"""技术因子（纯 pandas 向量化实现）。

每个因子函数：Callable[[pd.DataFrame], pd.Series]，输入含 open/high/low/close/volume 列的 DataFrame。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """RSI 相对强弱指标。Wilder 平滑法。"""
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder 平滑 = EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def ema(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """指数移动平均。"""
    return df["close"].ewm(span=period, adjust=False).mean()


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD 三列：dif / dea / hist。返回 DataFrame 而非 Series（多因子合一）。"""
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2
    return pd.DataFrame({"dif": dif, "dea": dea, "hist": hist}, index=df.index)


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ADX 趋势强度指标。返回 0~100 数列。"""
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff().clip(lower=0)
    down_move = (-low.diff()).clip(lower=0)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0)


def kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
    """KDJ 三列：K / D / J。"""
    low_n = df["low"].rolling(n).min()
    high_n = df["high"].rolling(n).max()
    rsv = (df["close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    k = rsv.ewm(alpha=1 / m1, adjust=False).mean().fillna(50)
    d = k.ewm(alpha=1 / m2, adjust=False).mean().fillna(50)
    j = 3 * k - 2 * d
    return pd.DataFrame({"K": k, "D": d, "J": j}, index=df.index)


def bollinger(df: pd.DataFrame, period: int = 20, k: float = 2.0) -> pd.DataFrame:
    """布林带：upper / mid / lower。"""
    mid = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    return pd.DataFrame({"upper": mid + k * std, "mid": mid, "lower": mid - k * std}, index=df.index)
