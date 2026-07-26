"""趋势类信号：组合 EMA / ADX 因子出买卖决策。"""

from __future__ import annotations

import pandas as pd

from eq.strategy import BUY, SELL, HOLD
from eq.strategy.factors.technical import adx, ema


def ema_cross(df: pd.DataFrame, fast: int = 5, slow: int = 20) -> pd.Series:
    """EMA 快线穿越慢线：金叉 BUY / 死叉 SELL / 否则 HOLD。

    Returns:
        pd.Series，index 同 df，值 ∈ {BUY, SELL, HOLD}
    """
    ema_f = ema(df, fast)
    ema_s = ema(df, slow)
    cross = ema_f - ema_s
    prev_cross = cross.shift(1)
    signal = pd.Series(HOLD, index=df.index, name="ema_cross")
    signal[(cross > 0) & (prev_cross <= 0)] = BUY
    signal[(cross < 0) & (prev_cross >= 0)] = SELL
    return signal


def adx_trend(df: pd.DataFrame, period: int = 14, threshold: float = 25.0) -> pd.Series:
    """ADX 趋势强度：ADX > threshold 且 close 上穿 20EMA → BUY，下穿 → SELL。"""
    adx_val = adx(df, period)
    ema20 = ema(df, 20)
    above = df["close"] > ema20
    # bool Series 做无填充的 shift 会引入 NaN 并把 dtype 退化成 object，
    # 之后的 `~prev_above` 直接 TypeError: bad operand type for unary ~: 'float'
    # ——adx_trend 此前在任何数据上都必崩。用 fill_value=False 保住 bool dtype。
    prev_above = above.shift(1, fill_value=False)
    signal = pd.Series(HOLD, index=df.index, name="adx_trend")
    strong = adx_val > threshold
    signal[strong & above & ~prev_above] = BUY
    signal[strong & ~above & prev_above] = SELL
    return signal
