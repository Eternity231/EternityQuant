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
    below = close < bands["lower"]
    entered_from_below = (close >= bands["lower"]) & (prev_close < bands["lower"])
    signal[entered_from_below] = BUY
    # 升破上轨后回落入上轨
    above = close > bands["upper"]
    entered_from_above = (close <= bands["upper"]) & (prev_close > bands["upper"])
    signal[entered_from_above] = SELL
    return signal
