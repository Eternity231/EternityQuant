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


# ---------------------------------------------------------------------------
# v0.27 扩充：原来只有 RSI/EMA/MACD/ADX/KDJ/布林 六个，
# 信号层只能做单指标交叉，做不了趋势强度过滤、波动率定仓、通道突破。
# 下面按「波动率 / 通道 / 动量 / 摆荡」四组补齐。
# ---------------------------------------------------------------------------


def true_range(df: pd.DataFrame) -> pd.Series:
    """真实波幅 TR = max(高-低, |高-前收|, |低-前收|)。ATR 与 Supertrend 的基础。"""
    prev_close = df["close"].shift()
    return pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """平均真实波幅（Wilder 平滑）。

    ATR 是**按波动率定仓和设止损**的基础：同样是 2%，高波动股票的
    2% 止损可能一天就被打掉，低波动股票的 2% 可能几周都碰不到。
    用 ATR 的倍数设止损才是可比的。
    """
    return true_range(df).ewm(alpha=1 / period, adjust=False).mean()


def natr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """归一化 ATR（ATR / 收盘价 × 100），跨标的可比的波动率。"""
    return atr(df, period) / df["close"].replace(0, np.nan) * 100


def realized_vol(df: pd.DataFrame, period: int = 20, annualize: bool = True) -> pd.Series:
    """已实现波动率：对数收益的滚动标准差。"""
    logret = np.log(df["close"] / df["close"].shift())
    vol = logret.rolling(period).std()
    return vol * np.sqrt(252) if annualize else vol


def donchian(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """唐奇安通道：upper/lower/mid。海龟交易法的核心，突破上轨做多。

    注意上下轨都 **shift(1)**——用「截至昨日」的 N 日高低点，
    否则今日最高价本身就在窗口里，"突破今日高点"永远不可能成立。
    """
    upper = df["high"].rolling(period).max().shift(1)
    lower = df["low"].rolling(period).min().shift(1)
    return pd.DataFrame({"upper": upper, "lower": lower, "mid": (upper + lower) / 2},
                        index=df.index)


def keltner(df: pd.DataFrame, period: int = 20, mult: float = 2.0) -> pd.DataFrame:
    """肯特纳通道：EMA ± mult × ATR。比布林带更平滑（用 ATR 而非标准差）。"""
    mid = df["close"].ewm(span=period, adjust=False).mean()
    band = atr(df, period) * mult
    return pd.DataFrame({"upper": mid + band, "mid": mid, "lower": mid - band},
                        index=df.index)


def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.DataFrame:
    """SuperTrend：ATR 通道 + 方向状态机。返回 ``line`` 与 ``trend``(+1/-1)。

    比裸均线交叉抗震荡：轨道只朝有利方向移动，价格站上/跌破才翻向。
    """
    hl2 = (df["high"] + df["low"]) / 2
    a = atr(df, period)
    upper_basic = hl2 + mult * a
    lower_basic = hl2 - mult * a
    close = df["close"].to_numpy(dtype="float64")
    ub = upper_basic.to_numpy(dtype="float64")
    lb = lower_basic.to_numpy(dtype="float64")
    n = len(df)
    final_ub = np.full(n, np.nan)
    final_lb = np.full(n, np.nan)
    trend = np.ones(n)
    for i in range(n):
        if i == 0 or np.isnan(ub[i]) or np.isnan(final_ub[i - 1]):
            final_ub[i], final_lb[i] = ub[i], lb[i]
            continue
        # 上轨只在「更低」或「上一根收盘已突破」时更新，下轨相反
        final_ub[i] = ub[i] if (ub[i] < final_ub[i - 1] or close[i - 1] > final_ub[i - 1]) else final_ub[i - 1]
        final_lb[i] = lb[i] if (lb[i] > final_lb[i - 1] or close[i - 1] < final_lb[i - 1]) else final_lb[i - 1]
        if trend[i - 1] == 1:
            trend[i] = -1 if close[i] < final_lb[i] else 1
        else:
            trend[i] = 1 if close[i] > final_ub[i] else -1
    line = np.where(trend == 1, final_lb, final_ub)
    return pd.DataFrame({"line": line, "trend": trend}, index=df.index)


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """顺势指标 CCI。±100 外视为强趋势区。"""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(period).mean()
    md = (tp - ma).abs().rolling(period).mean()
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """威廉指标 %R，取值 -100~0。低于 -80 超卖，高于 -20 超买。"""
    hh = df["high"].rolling(period).max()
    ll = df["low"].rolling(period).min()
    return -100 * (hh - df["close"]) / (hh - ll).replace(0, np.nan)


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """随机指标：%K / %D。"""
    ll = df["low"].rolling(k_period).min()
    hh = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)
    return pd.DataFrame({"K": k, "D": k.rolling(d_period).mean()}, index=df.index)


def roc(df: pd.DataFrame, period: int = 12) -> pd.Series:
    """变动率 ROC = (今收 / N 日前收 - 1) × 100。"""
    return df["close"].pct_change(period) * 100


def momentum(df: pd.DataFrame, period: int = 20, skip: int = 0) -> pd.Series:
    """时序动量。``skip`` 跳过最近几日（学术上跳过最近 1 个月能避开短期反转）。"""
    ref = df["close"].shift(skip)
    return ref / ref.shift(period) - 1


def zscore(df: pd.DataFrame, period: int = 20, col: str = "close") -> pd.Series:
    """滚动 z-score：(x - 均值) / 标准差。均值回归策略的核心。"""
    s = df[col]
    mean = s.rolling(period).mean()
    std = s.rolling(period).std()
    return (s - mean) / std.replace(0, np.nan)


def bollinger_bandwidth(df: pd.DataFrame, period: int = 20, k: float = 2.0) -> pd.Series:
    """布林带宽 (upper-lower)/mid。收窄到历史低位常是变盘前兆。"""
    b = bollinger(df, period, k)
    return (b["upper"] - b["lower"]) / b["mid"].replace(0, np.nan)


def trend_strength(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """趋势强度 0~100（就是 ADX，起个直白名字供市场状态判别用）。"""
    return adx(df, period)
