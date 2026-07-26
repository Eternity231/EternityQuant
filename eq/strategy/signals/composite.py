"""组合与择时信号（v0.27 新增）—— 策略层最缺的一块。

**原来薄在哪**

四个内置策略全是「单指标 + 交叉」：EMA 金叉、ADX 过滤、RSI 超买超卖、
布林突破。三个问题：

1. **没有组合**。单指标策略的 IC 天然很低，实盘靠的是多个弱信号叠加。
   原来想做「均线金叉 + 放量 + RSI 不超买」只能自己再写一个函数。
2. **没有择时**。趋势策略在震荡市亏钱、均值回归在单边市亏钱，这是结构性的。
   `eq backtest --sweep` 里 ema_cross 在茅台那段行情 -11%、rsi_reversal +15%，
   不是因为哪个策略更好，而是那段行情正好适合后者。**该做的是识别市场状态、
   切换策略**，而不是挑一个"最好的"。
3. **信号无强弱**，所以仓位只能满仓/空仓两档。

本模块给三件东西：

- :func:`vote`             多策略加权投票
- :func:`regime_adaptive`  按市场状态（趋势/震荡）切换策略
- :func:`filtered`         给任意策略加前置过滤条件（放量、趋势强度、流动性…）
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from eq.strategy import BUY, HOLD, SELL
from eq.strategy.factors.technical import (
    bollinger_bandwidth, natr, trend_strength, zscore,
)
from eq.strategy.signals.base import clip_score, score_to_signal, signal_to_score

SignalFunc = Callable[[pd.DataFrame], pd.Series]


def _as_score(fn: SignalFunc, df: pd.DataFrame) -> pd.Series:
    """跑一个策略并拿到分数：三态自动转分数，分数原样用。"""
    out = pd.Series(fn(df))
    if out.dtype == object or out.isin([BUY, SELL, HOLD]).any():
        return signal_to_score(out).reindex(df.index).fillna(0.0)
    return clip_score(out).reindex(df.index).fillna(0.0)


# ======================================================================
# 1) 多策略投票
# ======================================================================

def vote(
    df: pd.DataFrame,
    strategies: Sequence[SignalFunc],
    weights: Sequence[float] | None = None,
    buy_th: float = 0.35,
    sell_th: float = -0.35,
    *,
    min_agree: int = 0,
    as_score: bool = False,
) -> pd.Series:
    """多策略加权投票。

    Args:
        strategies: 子策略列表（三态或分数都行，自动统一成分数）
        weights: 权重，缺省等权。会归一化，所以传绝对值即可
        min_agree: 至少要有几个子策略**同向**才认。默认 0 = 不做这个约束。
            设成 2 就是「至少两票同意」，能有效压掉单一策略的噪声
        as_score: True 返回合成分数（便于再嵌套组合），False 返回三态

    Returns:
        三态信号（或分数）
    """
    if not strategies:
        raise ValueError("至少要传一个子策略")
    w = np.ones(len(strategies)) if weights is None else np.asarray(weights, dtype="float64")
    if len(w) != len(strategies):
        raise ValueError(f"权重个数 {len(w)} 与策略个数 {len(strategies)} 不一致")
    if w.sum() == 0:
        raise ValueError("权重之和不能为 0")
    w = w / np.abs(w).sum()

    scores = [_as_score(fn, df) for fn in strategies]
    mat = pd.concat(scores, axis=1)
    combined = clip_score((mat * w).sum(axis=1))

    if min_agree > 0:
        # 同向票数不足就压成中性——避免「一个策略给满分、其余全中性」也触发
        n_long = (mat > 0).sum(axis=1)
        n_short = (mat < 0).sum(axis=1)
        enough = ((combined > 0) & (n_long >= min_agree)) | ((combined < 0) & (n_short >= min_agree))
        combined = combined.where(enough, 0.0)

    if as_score:
        return combined.rename("vote_score")
    return score_to_signal(combined, buy_th, sell_th).rename("vote")


def make_vote(
    strategies: Sequence[SignalFunc],
    weights: Sequence[float] | None = None,
    **kwargs,
) -> SignalFunc:
    """把投票组合固化成一个 ``SignalFunc``，可直接丢给回测引擎。"""
    def _fn(df: pd.DataFrame) -> pd.Series:
        return vote(df, strategies, weights, **kwargs)

    names = "+".join(getattr(s, "__name__", "anon") for s in strategies)
    _fn.__name__ = f"vote({names})"
    return _fn


# ======================================================================
# 2) 市场状态识别 + 自适应切换
# ======================================================================

def market_regime(
    df: pd.DataFrame,
    adx_period: int = 14,
    trend_th: float = 25.0,
    range_th: float = 20.0,
) -> pd.Series:
    """判别市场状态：``trend`` / ``range`` / ``transition``。

    用 ADX 做主判据（>25 有趋势、<20 震荡），中间地带记 ``transition``。
    两个阈值留出缓冲带是为了避免在 25 附近反复横跳导致策略频繁切换。
    """
    adx_val = trend_strength(df, adx_period)
    regime = pd.Series("transition", index=df.index, name="regime", dtype=object)
    regime[adx_val >= trend_th] = "trend"
    regime[adx_val <= range_th] = "range"
    # 前面 NaN 的部分（指标未成形）当震荡处理，比当趋势保守
    regime[adx_val.isna()] = "range"
    return regime


def regime_adaptive(
    df: pd.DataFrame,
    trend_strategy: SignalFunc,
    range_strategy: SignalFunc,
    *,
    adx_period: int = 14,
    trend_th: float = 25.0,
    range_th: float = 20.0,
    transition: str = "flat",
) -> pd.Series:
    """按市场状态在两个策略之间切换。

    这是对「哪个策略最好」这个问题的正解：没有最好的策略，只有适合当前
    市场状态的策略。趋势市用趋势跟随，震荡市用均值回归。

    Args:
        transition: 过渡区（ADX 在两阈值之间）怎么办——
            ``flat`` 空仓观望（默认，最保守）| ``trend`` 用趋势策略 |
            ``range`` 用均值回归 | ``hold`` 维持前一个状态的信号
    """
    regime = market_regime(df, adx_period, trend_th, range_th)
    t_sig = pd.Series(trend_strategy(df)).reindex(df.index)
    r_sig = pd.Series(range_strategy(df)).reindex(df.index)

    out = pd.Series(HOLD, index=df.index, name="regime_adaptive", dtype=object)
    out[regime == "trend"] = t_sig[regime == "trend"]
    out[regime == "range"] = r_sig[regime == "range"]

    mid = regime == "transition"
    if transition == "trend":
        out[mid] = t_sig[mid]
    elif transition == "range":
        out[mid] = r_sig[mid]
    elif transition == "flat":
        # 从趋势/震荡切进过渡区的那一根发 SELL 清仓，之后保持 HOLD
        entering = mid & ~mid.shift(1, fill_value=False)
        out[entering] = SELL
    # transition == "hold" 时保持 HOLD，由引擎维持既有仓位
    return out


def make_regime_adaptive(trend_strategy: SignalFunc, range_strategy: SignalFunc,
                         **kwargs) -> SignalFunc:
    """固化成 ``SignalFunc``。"""
    def _fn(df: pd.DataFrame) -> pd.Series:
        return regime_adaptive(df, trend_strategy, range_strategy, **kwargs)

    _fn.__name__ = (f"regime({getattr(trend_strategy, '__name__', '?')}"
                    f"|{getattr(range_strategy, '__name__', '?')})")
    return _fn


# ======================================================================
# 3) 前置过滤器
# ======================================================================

def volume_filter(df: pd.DataFrame, period: int = 20, min_ratio: float = 1.0) -> pd.Series:
    """量能闸门：当日量 / N 日均量 ≥ min_ratio 才放行。无量的突破多半是假突破。"""
    avg = df["volume"].rolling(period).mean().shift(1)
    return (df["volume"] / avg.replace(0, np.nan)) >= min_ratio


def volatility_filter(df: pd.DataFrame, period: int = 14,
                      min_natr: float = 0.5, max_natr: float = 15.0) -> pd.Series:
    """波动率闸门：太死（没波动赚不到钱）和太疯（止损必被打）的都过滤掉。"""
    v = natr(df, period)
    return (v >= min_natr) & (v <= max_natr)


def squeeze_filter(df: pd.DataFrame, period: int = 20, quantile: float = 0.3) -> pd.Series:
    """收口闸门：只在布林带宽处于近期低位（蓄势）时放行，捕捉变盘。"""
    bw = bollinger_bandwidth(df, period)
    th = bw.rolling(period * 3, min_periods=period).quantile(quantile)
    return bw <= th


def not_overextended(df: pd.DataFrame, period: int = 20, max_z: float = 2.5) -> pd.Series:
    """乖离闸门：价格偏离均值太远时不再追（追高是散户最大的亏损来源之一）。"""
    return zscore(df, period).abs() <= max_z


def filtered(
    df: pd.DataFrame,
    strategy: SignalFunc,
    filters: Sequence[Callable[[pd.DataFrame], pd.Series]],
    *,
    apply_to: str = "buy",
) -> pd.Series:
    """给策略加前置过滤：条件不满足时把信号压成 HOLD。

    Args:
        apply_to: ``buy`` 只过滤买入信号（默认——离场信号不该被过滤掉，
            否则该跑的时候跑不掉）| ``both`` 买卖都过滤
    """
    sig = pd.Series(strategy(df)).reindex(df.index)
    if not filters:
        return sig
    ok = pd.Series(True, index=df.index)
    for f in filters:
        ok &= pd.Series(f(df)).reindex(df.index).fillna(False).astype(bool)
    blocked = ~ok
    out = sig.copy()
    if apply_to == "both":
        out[blocked] = HOLD
    else:
        out[blocked & (sig == BUY)] = HOLD
    return out.rename("filtered")


def make_filtered(strategy: SignalFunc,
                  filters: Sequence[Callable[[pd.DataFrame], pd.Series]],
                  **kwargs) -> SignalFunc:
    """固化成 ``SignalFunc``。"""
    def _fn(df: pd.DataFrame) -> pd.Series:
        return filtered(df, strategy, filters, **kwargs)

    _fn.__name__ = f"filtered({getattr(strategy, '__name__', '?')})"
    return _fn
