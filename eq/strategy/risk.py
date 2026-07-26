"""仓位管理与风控（v0.27 新增）—— 策略层第二大缺口。

**原来的问题**

回测引擎把信号解释成「BUY = 满仓、SELL = 空仓」，只有两档。这意味着：

1. **没有仓位管理**。一只年化波动 15% 的银行股和一只波动 60% 的题材股，
   同样满仓，组合风险完全由后者主导。专业做法是**按波动率定仓**：
   波动大的少买，让每笔交易承担的风险大致相等。
2. **没有止损**。策略只在出现反向信号时才离场。趋势策略在趋势反转时
   往往已经回撤很多——`eq backtest --sweep` 里 ema_cross 最大回撤 -18.8%
   就是这么来的。
3. **没有回撤熔断**。连续亏损时不会自动降仓。

本模块把「信号」和「仓位」拆开：信号只管方向，仓位由这里决定。

- :func:`fixed_fraction`      固定比例
- :func:`volatility_target`   波动率目标定仓（组合风险均衡的标准做法）
- :func:`atr_risk_size`       按「单笔最大亏损 = 账户的 x%」反推仓位
- :func:`apply_stops`         ATR 止损 / 跟踪止损 / 时间止损
- :func:`drawdown_throttle`   回撤熔断降仓
- :func:`build_positions`     串起来：信号 → 目标仓位序列（0~1）

产出的目标仓位序列可以直接喂给回测引擎（v0.27 起引擎支持连续仓位）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eq.strategy import BUY, SELL
from eq.strategy.factors.technical import atr, realized_vol

TRADING_DAYS = 252


# ======================================================================
# 仓位大小
# ======================================================================

def fixed_fraction(df: pd.DataFrame, fraction: float = 1.0) -> pd.Series:
    """固定比例仓位（原来的行为等价于 fraction=1.0）。"""
    return pd.Series(float(np.clip(fraction, 0.0, 1.0)), index=df.index, name="size")


def volatility_target(
    df: pd.DataFrame,
    target_vol: float = 0.20,
    period: int = 20,
    max_leverage: float = 1.0,
    min_size: float = 0.05,
) -> pd.Series:
    """波动率目标定仓：``仓位 = 目标波动 / 标的年化波动``。

    这是机构做组合最常用的定仓方式。直觉：想让组合年化波动稳定在 20%，
    那么波动 40% 的标的就只能买半仓，波动 10% 的可以满仓。
    效果是**每笔交易的风险贡献大致相等**，而不是被高波动标的绑架。

    Args:
        target_vol: 目标年化波动率（0.20 = 20%）
        max_leverage: 仓位上限。默认 1.0 不加杠杆——低波动标的算出来
            可能 >1，散户账户没杠杆就该截断
        min_size: 仓位下限，避免算出 0.1% 这种没意义的仓位
    """
    vol = realized_vol(df, period, annualize=True)
    size = target_vol / vol.replace(0, np.nan)
    return size.clip(min_size, max_leverage).fillna(min_size).rename("size")


def atr_risk_size(
    df: pd.DataFrame,
    risk_per_trade: float = 0.02,
    atr_mult: float = 2.0,
    period: int = 14,
    max_leverage: float = 1.0,
    min_size: float = 0.05,
) -> pd.Series:
    """按「单笔最大亏损占账户 x%」反推仓位（海龟法则的定仓方式）。

    止损设在入场价下方 ``atr_mult × ATR``，那么每 1 元仓位的潜在亏损是
    ``atr_mult × ATR / 价格``。要让这笔亏损等于账户的 ``risk_per_trade``，
    仓位就是 ``risk_per_trade / (atr_mult × ATR / 价格)``。

    和 :func:`volatility_target` 的区别：这个直接锚定**止损距离**，
    和你实际设的止损位一致；波动率目标锚定的是**收益波动**。
    """
    a = atr(df, period)
    stop_dist_pct = (atr_mult * a / df["close"].replace(0, np.nan))
    size = risk_per_trade / stop_dist_pct.replace(0, np.nan)
    return size.clip(min_size, max_leverage).fillna(min_size).rename("size")


def score_scaled_size(score: pd.Series, base: float = 1.0,
                      min_size: float = 0.0) -> pd.Series:
    """按信号分数强弱定仓：``|score| × base``。

    配合 :mod:`eq.strategy.signals.base` 的分数信号用——
    刚触发阈值和强烈看多，仓位不该一样。
    """
    s = pd.Series(score).abs().clip(0.0, 1.0) * base
    return s.clip(min_size, base).rename("size")


# ======================================================================
# 止损
# ======================================================================

def apply_stops(
    df: pd.DataFrame,
    position: pd.Series,
    *,
    atr_mult: float = 2.5,
    atr_period: int = 14,
    trailing: bool = True,
    max_hold_bars: int = 0,
    take_profit_mult: float = 0.0,
) -> pd.DataFrame:
    """对目标仓位序列施加止损，返回带 ``position`` / ``exit_reason`` 的表。

    逐 bar 状态机（止损天然是路径依赖的，没法向量化）：

    - **ATR 止损**：跌破 ``入场价 - atr_mult × 入场时ATR`` 清仓
    - **跟踪止损**（``trailing=True``）：止损位随持仓期最高价上移，只涨不跌
    - **时间止损**（``max_hold_bars>0``）：持有超过 N 根还没走出来就撤
    - **止盈**（``take_profit_mult>0``）：涨到 ``入场价 + mult × ATR`` 落袋

    止损触发后当根清仓，且**必须等下一次信号从 0 变正才重新入场**——
    否则目标仓位一直是正的，下一根马上又买回来，止损就白设了。
    """
    pos = pd.Series(position).astype("float64").reindex(df.index).fillna(0.0)
    a = atr(df, atr_period)
    close = df["close"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    high = df["high"].to_numpy(dtype="float64")
    atr_arr = a.to_numpy(dtype="float64")
    tgt = pos.to_numpy(dtype="float64")

    n = len(df)
    out = np.zeros(n)
    reason = np.array([""] * n, dtype=object)

    in_pos = False
    entry_price = stop_price = tp_price = np.nan
    peak = np.nan
    held = 0
    blocked = False          # 止损后锁定，等目标仓位归零再解锁

    for i in range(n):
        want = tgt[i]
        if in_pos:
            held += 1
            peak = max(peak, high[i]) if not np.isnan(peak) else high[i]
            if trailing and not np.isnan(atr_arr[i]):
                stop_price = max(stop_price, peak - atr_mult * atr_arr[i])

            hit = ""
            if not np.isnan(stop_price) and low[i] <= stop_price:
                hit = "stop_loss"
            elif take_profit_mult > 0 and not np.isnan(tp_price) and high[i] >= tp_price:
                hit = "take_profit"
            elif max_hold_bars > 0 and held >= max_hold_bars:
                hit = "time_stop"
            elif want <= 0:
                hit = "signal_exit"

            if hit:
                out[i] = 0.0
                reason[i] = hit
                in_pos = False
                held = 0
                blocked = hit in ("stop_loss", "take_profit", "time_stop")
                continue
            out[i] = want if want > 0 else 0.0
            continue

        # 空仓状态
        if want <= 0:
            blocked = False       # 目标仓位归零 → 解锁，允许下次重新入场
            out[i] = 0.0
            continue
        if blocked:
            out[i] = 0.0
            continue
        # 开仓
        in_pos = True
        held = 0
        entry_price = close[i]
        peak = high[i]
        cur_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else 0.0
        stop_price = entry_price - atr_mult * cur_atr
        tp_price = (entry_price + take_profit_mult * cur_atr) if take_profit_mult > 0 else np.nan
        out[i] = want

    return pd.DataFrame({"position": out, "exit_reason": reason}, index=df.index)


# ======================================================================
# 回撤熔断
# ======================================================================

def drawdown_throttle(
    equity_or_close: pd.Series,
    position: pd.Series,
    *,
    warn_dd: float = 0.10,
    halt_dd: float = 0.20,
    warn_scale: float = 0.5,
) -> pd.Series:
    """按回撤降仓：回撤超 ``warn_dd`` 砍到 ``warn_scale``，超 ``halt_dd`` 清仓。

    这是把「亏到一定程度就停手」制度化。注意用的是**策略自身的**
    权益回撤，不是标的价格回撤——所以实盘要传入组合权益曲线。
    """
    s = pd.Series(equity_or_close).astype("float64")
    dd = (s / s.cummax() - 1.0).fillna(0.0)
    scale = pd.Series(1.0, index=s.index)
    scale[dd <= -warn_dd] = warn_scale
    scale[dd <= -halt_dd] = 0.0
    return (pd.Series(position).astype("float64") * scale).clip(0.0, 1.0).rename("position")


# ======================================================================
# 串起来
# ======================================================================

def build_positions(
    df: pd.DataFrame,
    signal: pd.Series,
    *,
    sizing: str = "fixed",
    sizing_kwargs: dict | None = None,
    stops: bool = True,
    stop_kwargs: dict | None = None,
) -> pd.Series:
    """信号 → 目标仓位序列（0~1），可直接喂回测引擎。

    Args:
        signal: 三态信号或分数序列
        sizing: ``fixed`` | ``vol_target`` | ``atr_risk`` | ``score``
        stops: 是否施加 :func:`apply_stops`

    Returns:
        逐 bar 的目标仓位（0~1）。引擎见到数值型序列会当仓位直接用。
    """
    sig = pd.Series(signal).reindex(df.index)
    is_ternary = sig.dtype == object or sig.isin([BUY, SELL]).any()

    # 1) 方向：三态 ffill 成持仓状态；分数则正为持有
    if is_ternary:
        raw = pd.Series(np.nan, index=df.index, dtype="float64")
        raw[sig == BUY] = 1.0
        raw[sig == SELL] = 0.0
        direction = raw.ffill().fillna(0.0)
        score = direction
    else:
        score = pd.Series(sig).astype("float64").fillna(0.0)
        direction = (score > 0).astype("float64")

    # 2) 大小
    kw = dict(sizing_kwargs or {})
    if sizing == "vol_target":
        size = volatility_target(df, **kw)
    elif sizing == "atr_risk":
        size = atr_risk_size(df, **kw)
    elif sizing == "score":
        size = score_scaled_size(score, **kw)
    elif sizing == "fixed":
        size = fixed_fraction(df, **kw)
    else:
        raise ValueError(f"未知定仓方式 {sizing}，可选 fixed/vol_target/atr_risk/score")

    pos = (direction * size).clip(0.0, 1.0)

    # 3) 止损
    if stops:
        pos = apply_stops(df, pos, **(stop_kwargs or {}))["position"]
    return pos.rename("position")


def make_managed(signal_fn, **kwargs):
    """把「信号 + 风控」打包成一个可直接回测的 ``SignalFunc``。"""
    def _fn(df: pd.DataFrame) -> pd.Series:
        return build_positions(df, signal_fn(df), **kwargs)

    _fn.__name__ = f"managed({getattr(signal_fn, '__name__', '?')})"
    return _fn
