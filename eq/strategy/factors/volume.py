"""成交量因子。"""

from __future__ import annotations

import pandas as pd


def obv(df: pd.DataFrame) -> pd.Series:
    """On Balance Volume 量能潮。"""
    direction = (df["close"].diff() > 0).astype(int) - (df["close"].diff() < 0).astype(int)
    return (direction * df["volume"]).cumsum()


def volume_ratio(df: pd.DataFrame, period: int = 5) -> pd.Series:
    """量比：当日量 / 前 period 日均量。"""
    avg = df["volume"].rolling(period).mean().shift(1)
    return df["volume"] / avg.replace(0, pd.NA)
