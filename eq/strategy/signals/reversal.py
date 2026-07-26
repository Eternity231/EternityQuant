"""反转类信号：组合 RSI / 布林因子出买卖决策。"""

from __future__ import annotations

import pandas as pd

from eq.strategy import BUY, SELL, HOLD
from eq.strategy.factors.technical import bollinger, rsi


def rsi_reversal(df: pd.DataFrame, period: int = 14, oversold: float = 30.0, overbought: float = 70.0) -> pd.Series:
    """RSI 超卖回升 BUY / 超买回落 SELL。"""
    rsi_val = rsi(df, period)
    prev = rsi_val.shift(1)
    signal = pd.Series(HOLD, index=df.index, name="rsi_reversal")
    signal[(rsi_val > oversold) & (prev <= oversold)] = BUY
    signal[(rsi_val < overbought) & (prev >= overbought)] = SELL
    return signal


def bollinger_break(df: pd.DataFrame, period: int = 20, k: float = 2.0) -> pd.Series:
    """布林带突破：跌破下轨后回升入下轨 BUY / 升破上轨后回落入上轨 SELL。"""
    bands = bollinger(df, period, k)
    close = df["close"]
    prev_close = close.shift(1)
    signal = pd.Series(HOLD, index=df.index, name="bollinger_break")
    # 跌破下轨后回升入下轨
    entered_from_below = (close >= bands["lower"]) & (prev_close < bands["lower"])
    signal[entered_from_below] = BUY
    # 升破上轨后回落入上轨
    entered_from_above = (close <= bands["upper"]) & (prev_close > bands["upper"])
    signal[entered_from_above] = SELL
    return signal


# ---------------------------------------------------------------------------
# v0.27 扩充：原来只有 RSI 和布林两个反转信号
# ---------------------------------------------------------------------------

def zscore_reversion(df: pd.DataFrame, period: int = 20,
                     entry: float = 2.0, exit_z: float = 0.5) -> pd.Series:
    """z-score 均值回归：跌到 -entry σ 买、回到 -exit_z σ 卖。

    这是均值回归最直白的形式，也是震荡市里最稳的一类。
    进出阈值分开设（进场要极端、出场要及时）——用同一个阈值会在
    阈值附近反复穿越，换手极高。
    """
    from eq.strategy.factors.technical import zscore

    z = zscore(df, period)
    prev = z.shift(1)
    sig = pd.Series(HOLD, index=df.index, name="zscore_reversion")
    sig[(z <= -entry) & (prev > -entry)] = BUY
    sig[(z >= -exit_z) & (prev < -exit_z)] = SELL
    return sig


def kdj_cross(df: pd.DataFrame, n: int = 9, oversold: float = 20.0,
              overbought: float = 80.0) -> pd.Series:
    """KDJ 金叉/死叉，且只在超卖区金叉才买、超买区死叉才卖。"""
    from eq.strategy.factors.technical import kdj

    k = kdj(df, n=n)
    kk, dd = k["K"], k["D"]
    cross_up = (kk > dd) & (kk.shift(1) <= dd.shift(1))
    cross_dn = (kk < dd) & (kk.shift(1) >= dd.shift(1))
    sig = pd.Series(HOLD, index=df.index, name="kdj_cross")
    sig[cross_up & (dd < oversold)] = BUY
    sig[cross_dn & (dd > overbought)] = SELL
    return sig


def cci_reversal(df: pd.DataFrame, period: int = 20, level: float = 100.0) -> pd.Series:
    """CCI 从超卖区（<-level）回升买入，从超买区（>+level）回落卖出。"""
    from eq.strategy.factors.technical import cci

    c = cci(df, period)
    prev = c.shift(1)
    sig = pd.Series(HOLD, index=df.index, name="cci_reversal")
    sig[(c > -level) & (prev <= -level)] = BUY
    sig[(c < level) & (prev >= level)] = SELL
    return sig


def rsi_score(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """RSI 的**分数版**：RSI 50 → 0，RSI 0 → +1（极度超卖看多），RSI 100 → -1。

    注意符号是反的——均值回归里超卖是看多信号。
    """
    from eq.strategy.factors.technical import rsi
    from eq.strategy.signals.base import clip_score

    return clip_score((50.0 - rsi(df, period)) / 30.0).rename("rsi_score")


def reversion_composite(df: pd.DataFrame, buy_th: float = 0.35,
                        sell_th: float = -0.35) -> pd.Series:
    """RSI + z-score 两个反转分数合成，作为「均值回归」这一大类的代表策略。"""
    from eq.strategy.factors.technical import zscore
    from eq.strategy.signals.base import clip_score, score_to_signal

    s = (rsi_score(df) + clip_score(-zscore(df, 20) / 2.0)) / 2.0
    return score_to_signal(clip_score(s), buy_th, sell_th).rename("reversion_composite")
