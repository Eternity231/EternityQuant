"""次日高点预测与限价止盈（v0.31 新增）。

**要做的事**

T 日收盘买入，T+1 日卖出（A 股 T+1 规则下这是最短的合法持有期）。
目标是尽量卖在 T+1 的高点附近。

**必须先破除的幻觉**

「按第二天最高价卖出」不是策略，是**未来函数**——你事先不知道高点在哪。
能真正执行的只有一种形式：**提前挂一个限价单**。于是问题变成

    该把限价挂在买入价上方多少？

挂太高摸不到（只能收盘平仓），挂太低白白让出利润。这是个可以量化的
权衡，本模块就是干这个的：

- :func:`build_dataset`     造特征 + 三个标签（MFE / MAE / 收盘收益）
- :func:`simulate_limit`    **正确建模限价成交**的回测（含跳空、涨跌停、T+1）
- :func:`scan_targets`      扫不同限价档位，给出「成交率 vs 收益」曲线
- :func:`baseline_stats`    次日 MFE/MAE 的分布——先看清楚天花板在哪

**几个容易忽略的现实约束**

1. **T+1**：T 日买、T+1 才能卖。所以买入必须发生在 T 日（尾盘），
   不能是 T+1 开盘。
2. **跳空高开**：若 T+1 开盘价已高于限价，实际成交在开盘价（比限价更好）。
3. **涨停**：涨停板上卖不出去（封死），这天只能继续持有。
4. **成本**：一天一个来回，印花税 + 最低佣金按 :mod:`eq.backtest.cost` 算。
   高频短线正是成本最致命的场景。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from eq.backtest.cost import CostModel, for_market

logger = logging.getLogger(__name__)


# ======================================================================
# 1) 先看清楚天花板：次日 MFE / MAE 的分布
# ======================================================================

def baseline_stats(bars: pd.DataFrame) -> dict[str, Any]:
    """T 日收盘买入后，T+1 日的三个基本分布。

    - **MFE**（最大有利偏移）= ``high_{T+1} / close_T - 1``：限价止盈的天花板
    - **MAE**（最大不利偏移）= ``low_{T+1} / close_T - 1``：这一天最多亏多少
    - **收盘收益** = ``close_{T+1} / close_T - 1``：什么都不做的基准

    看这三个数就知道「次日高点」这件事值不值得做：
    如果 MFE 中位数只有 1%，而一个来回成本就 0.3%，那空间非常有限。
    """
    c = bars["close"]
    mfe = bars["high"].shift(-1) / c - 1
    mae = bars["low"].shift(-1) / c - 1
    ret = c.shift(-1) / c - 1
    op = bars["open"].shift(-1) / c - 1
    valid = mfe.notna() & mae.notna() & ret.notna()
    mfe, mae, ret, op = mfe[valid], mae[valid], ret[valid], op[valid]
    if len(mfe) == 0:
        return {"n": 0}
    return {
        "n": int(len(mfe)),
        "mfe_mean": float(mfe.mean()), "mfe_median": float(mfe.median()),
        "mfe_p25": float(mfe.quantile(0.25)), "mfe_p75": float(mfe.quantile(0.75)),
        "mae_mean": float(mae.mean()), "mae_median": float(mae.median()),
        "close_ret_mean": float(ret.mean()), "close_ret_median": float(ret.median()),
        "overnight_mean": float(op.mean()),   # 隔夜跳空
        "pct_up_close": float((ret > 0).mean()),
        # 关键比值：上行空间 / 下行空间。<1 说明次日的风险回报天然不利
        "mfe_over_mae": float(mfe.median() / abs(mae.median())) if mae.median() else float("nan"),
    }


# ======================================================================
# 2) 特征与标签
# ======================================================================

def build_dataset(bars: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """造「次日高点预测」的特征表 + 标签。

    **所有特征都只用 T 日及之前的数据**（T 日收盘买入，所以 T 日收盘价可用）。

    标签：
        ``y_mfe``  次日最大有利偏移（要预测的主目标）
        ``y_mae``  次日最大不利偏移
        ``y_close`` 次日收盘收益（基准）
    """
    from eq.strategy.factors.technical import atr, natr, rsi, zscore

    df = bars
    c, h, low_, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"]
    out = pd.DataFrame(index=df.index)

    # --- 波动率类：次日振幅的最强预测因子就是近期振幅 ---
    out["natr14"] = natr(df, 14)
    out["atr_ratio"] = atr(df, 5) / atr(df, 20).replace(0, np.nan)
    out["range_pct"] = (h - low_) / c.replace(0, np.nan)
    out["range_ma5"] = out["range_pct"].rolling(5).mean()
    out["range_ma20"] = out["range_pct"].rolling(20).mean()
    out["range_expand"] = out["range_ma5"] / out["range_ma20"].replace(0, np.nan)

    # --- 当日形态：收盘在当日区间的位置（收在高位往往次日冲高） ---
    rng = (h - low_).replace(0, np.nan)
    out["close_pos"] = (c - low_) / rng
    out["upper_shadow"] = (h - np.maximum(c, o)) / rng
    out["lower_shadow"] = (np.minimum(c, o) - low_) / rng
    out["body"] = (c - o) / c.replace(0, np.nan)

    # --- 动量 / 位置 ---
    for p in (1, 3, 5, 10):
        out[f"ret{p}"] = c.pct_change(p)
    out["rsi14"] = rsi(df, 14)
    out["z20"] = zscore(df, 20)
    out["dist_ma20"] = c / c.rolling(20).mean() - 1

    # --- 量能：放量往往伴随次日更大波动 ---
    vma = v.rolling(20).mean()
    out["vol_ratio"] = v / vma.replace(0, np.nan)
    out["vol_ratio5"] = v.rolling(5).mean() / vma.replace(0, np.nan)

    # --- 隔夜/日内拆分 ---
    out["overnight"] = o / c.shift(1).replace(0, np.nan) - 1
    out["intraday"] = c / o.replace(0, np.nan) - 1

    # --- 标签 ---
    out["y_mfe"] = h.shift(-1) / c - 1
    out["y_mae"] = low_.shift(-1) / c - 1
    out["y_close"] = c.shift(-1) / c - 1
    out["entry_close"] = c
    if symbol:
        out["symbol"] = symbol
    return out.replace([np.inf, -np.inf], np.nan)


# ======================================================================
# 3) 限价止盈的正确建模
# ======================================================================

@dataclass
class LimitExitResult:
    trades: pd.DataFrame
    stats: dict[str, Any]

    def summary(self) -> str:
        s = self.stats
        return (f"{s['n_trades']} 笔  成交率 {s['fill_rate']:.1%}  "
                f"净收益/笔 {s['mean_net']:+.3%}  中位 {s['median_net']:+.3%}  "
                f"胜率 {s['win_rate']:.1%}  年化 {s['annualized']:+.2%}")


def simulate_limit(
    bars: pd.DataFrame,
    target_pct: float | pd.Series,
    *,
    entry: pd.Series | None = None,
    costs: CostModel | None = None,
    stop_pct: float | None = None,
    limit_band: float = 0.10,
) -> LimitExitResult:
    """T 日收盘买入、T+1 日限价止盈的**可执行**回测。

    成交规则（按真实撮合顺序）：

    1. T+1 开盘价 ≥ 限价 → 以**开盘价**成交（跳空高开，比限价更好）
    2. 否则 T+1 最高价 ≥ 限价 → 以**限价**成交
    3. 都没摸到 → T+1 **收盘价**平仓
    4. 若 T+1 涨停封板（最高价 = 收盘价 = 涨停价）→ 卖不出去，按收盘价计
       （实际会更好，这里保守处理）

    Args:
        target_pct: 限价相对买入价的涨幅。可以是常数，也可以是逐日的
            Series（由模型预测出来的动态目标）
        entry: 布尔序列，哪些 T 日买入。缺省全部买入
        stop_pct: 可选的日内止损（低于买入价这么多就割）。**注意**：
            止损和止盈同一天都触发时无法判断先后，这里保守地按**先止损**处理

    Returns:
        :class:`LimitExitResult`，含逐笔明细和汇总统计
    """
    costs = costs or for_market("A")
    c = bars["close"]
    nxt_o = bars["open"].shift(-1)
    nxt_h = bars["high"].shift(-1)
    nxt_l = bars["low"].shift(-1)
    nxt_c = bars["close"].shift(-1)

    tgt = (pd.Series(target_pct, index=bars.index)
           if np.isscalar(target_pct) else pd.Series(target_pct).reindex(bars.index))
    ent = (pd.Series(True, index=bars.index) if entry is None
           else pd.Series(entry).reindex(bars.index).fillna(False).astype(bool))

    ok = ent & c.notna() & nxt_o.notna() & nxt_h.notna() & nxt_c.notna() & tgt.notna()
    # T 日涨停封板买不进（简化：收盘涨幅≥9.9% 视为封板）
    limit_up_today = (c / c.shift(1) - 1) >= limit_band * 0.99
    ok &= ~limit_up_today.fillna(False)
    if not ok.any():
        return LimitExitResult(pd.DataFrame(), {"n_trades": 0, "fill_rate": 0.0,
                                                "mean_net": 0.0, "median_net": 0.0,
                                                "win_rate": 0.0, "annualized": 0.0})

    idx = bars.index[ok]
    entry_px = c[ok].to_numpy()
    limit_px = entry_px * (1 + tgt[ok].to_numpy())
    o1, h1, l1, c1 = (nxt_o[ok].to_numpy(), nxt_h[ok].to_numpy(),
                      nxt_l[ok].to_numpy(), nxt_c[ok].to_numpy())

    exit_px = np.where(o1 >= limit_px, o1,                      # 跳空高开
                       np.where(h1 >= limit_px, limit_px, c1))  # 摸到 / 没摸到
    filled = (o1 >= limit_px) | (h1 >= limit_px)
    reason = np.where(o1 >= limit_px, "gap_open",
                      np.where(h1 >= limit_px, "limit", "close"))

    if stop_pct is not None and stop_pct > 0:
        stop_px = entry_px * (1 - stop_pct)
        hit_stop = l1 <= stop_px
        # 同日既触止损又触止盈时无法判断先后，保守按先止损
        exit_px = np.where(hit_stop, np.minimum(stop_px, o1), exit_px)
        reason = np.where(hit_stop, "stop", reason)
        filled = filled & ~hit_stop

    gross = exit_px / entry_px - 1
    # 成本：一买一卖，按单笔金额算（最低佣金对小额单尤其致命）
    notional = 10_000.0    # 以 1 万元一笔估算费率；金额敏感度另见 cost 模块
    fee = costs.round_trip_ratio(notional) + 2 * costs.slippage_rate
    net = gross - fee

    trades = pd.DataFrame({
        "date": idx, "entry": entry_px, "limit": limit_px, "exit": exit_px,
        "gross": gross, "net": net, "filled": filled, "reason": reason,
        "mfe": h1 / entry_px - 1, "mae": l1 / entry_px - 1,
        "close_ret": c1 / entry_px - 1,
    }).set_index("date")

    n = len(trades)
    years = max((idx[-1] - idx[0]).days / 365.25, 1e-9) if n > 1 else 1e-9
    cum = float(np.prod(1 + net))
    stats = {
        "n_trades": n,
        "fill_rate": float(filled.mean()),
        "mean_gross": float(gross.mean()), "mean_net": float(net.mean()),
        "median_net": float(np.median(net)),
        "win_rate": float((net > 0).mean()),
        "total_return": cum - 1,
        "annualized": cum ** (1 / years) - 1 if cum > 0 else -1.0,
        "cost_per_trade": fee,
        # 捕获率：实际拿到的涨幅 / 理论最大涨幅（MFE）。上限 100%
        "capture_rate": float(np.mean(np.clip(gross, 0, None)
                                      / np.clip(trades["mfe"].to_numpy(), 1e-9, None))),
    }
    return LimitExitResult(trades, stats)


def scan_targets(
    bars: pd.DataFrame,
    targets: list[float] | None = None,
    *,
    entry: pd.Series | None = None,
    costs: CostModel | None = None,
    stop_pct: float | None = None,
) -> pd.DataFrame:
    """扫不同限价档位，给出「成交率 ↔ 收益」的权衡曲线。

    这是本模块最实用的函数：直接看出你的标的上，限价该挂多高。
    """
    targets = targets or [0.003, 0.005, 0.008, 0.010, 0.015, 0.020, 0.030, 0.050]
    rows = []
    for t in targets:
        r = simulate_limit(bars, t, entry=entry, costs=costs, stop_pct=stop_pct)
        if r.stats["n_trades"] == 0:
            continue
        rows.append({"限价档": f"+{t:.1%}", "target": t, **r.stats})
    return pd.DataFrame(rows)


def format_baseline(stats: dict[str, Any], title: str = "") -> str:
    """把 :func:`baseline_stats` 排成人话。"""
    if not stats.get("n"):
        return "\n样本不足\n"
    s = stats
    lines = [f"\n次日高低点分布{('（' + title + '）') if title else ''}  样本 {s['n']} 天\n"]
    lines.append("  次日最高 / 今收 - 1（MFE，限价止盈的天花板）")
    lines.append(f"    均值 {s['mfe_mean']:+.3%}   中位 {s['mfe_median']:+.3%}"
                 f"   四分位 [{s['mfe_p25']:+.3%}, {s['mfe_p75']:+.3%}]")
    lines.append("  次日最低 / 今收 - 1（MAE，当天最多亏多少）")
    lines.append(f"    均值 {s['mae_mean']:+.3%}   中位 {s['mae_median']:+.3%}")
    lines.append("  次日收盘收益（什么都不做的基准）")
    lines.append(f"    均值 {s['close_ret_mean']:+.3%}   中位 {s['close_ret_median']:+.3%}"
                 f"   上涨占比 {s['pct_up_close']:.1%}")
    lines.append(f"  隔夜跳空均值 {s['overnight_mean']:+.3%}")
    lines.append(f"  上行/下行空间比（MFE中位 / |MAE中位|） {s['mfe_over_mae']:.2f}")
    return "\n".join(lines) + "\n"
