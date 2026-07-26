"""突破/趋势跟随类信号（v0.27 新增）。

原来的趋势策略只有 EMA 交叉和 ADX 过滤两个，都是「均线穿越」这一个套路，
在震荡市里会被反复打脸。这里补三类结构不同的：

- **通道突破**（唐奇安 / 肯特纳）：只在价格创出 N 日新高时进场，天然过滤震荡
- **SuperTrend**：ATR 自适应轨道 + 方向状态机，轨道只朝有利方向移动
- **波动率突破**：以开盘价为基准、按前期波幅设触发带（Dual Thrust 思路）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eq.strategy import BUY, HOLD, SELL
from eq.strategy.factors.technical import (
    atr, donchian, keltner, supertrend, trend_strength,
)
from eq.strategy.signals.base import clip_score, score_to_signal


def donchian_breakout(df: pd.DataFrame, entry: int = 20, exit_period: int = 10) -> pd.Series:
    """唐奇安通道突破（海龟交易法）：收盘创 ``entry`` 日新高进场，
    跌破 ``exit_period`` 日新低离场。

    离场窗口比进场窗口短，是海龟法则的关键：进场要苛刻（确认趋势），
    离场要敏感（保住利润）。用同一个窗口进出会把利润全还回去。
    """
    up = donchian(df, entry)["upper"]
    dn = donchian(df, exit_period)["lower"]
    close = df["close"]
    sig = pd.Series(HOLD, index=df.index, name="donchian_breakout")
    sig[(close > up) & up.notna()] = BUY
    sig[(close < dn) & dn.notna()] = SELL
    return sig


def keltner_breakout(df: pd.DataFrame, period: int = 20, mult: float = 2.0) -> pd.Series:
    """肯特纳通道突破：站上上轨做多，跌破中轨离场。

    比布林带突破稳：布林带宽度用标准差，暴涨时带宽被自己撑大反而不触发；
    肯特纳用 ATR，对跳空更敏感。
    """
    k = keltner(df, period, mult)
    close = df["close"]
    sig = pd.Series(HOLD, index=df.index, name="keltner_breakout")
    sig[(close > k["upper"]) & k["upper"].notna()] = BUY
    sig[(close < k["mid"]) & k["mid"].notna()] = SELL
    return sig


def supertrend_follow(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.Series:
    """SuperTrend 翻向即进出场。趋势市表现好，震荡市靠 mult 拉大轨道减少假信号。"""
    st = supertrend(df, period, mult)
    trend = st["trend"]
    prev = trend.shift(1)
    sig = pd.Series(HOLD, index=df.index, name="supertrend_follow")
    sig[(trend == 1) & (prev == -1)] = BUY
    sig[(trend == -1) & (prev == 1)] = SELL
    return sig


def volatility_breakout(df: pd.DataFrame, period: int = 20, k: float = 0.5) -> pd.Series:
    """波动率突破（Dual Thrust 思路的日线版）。

    以今日开盘价为基准，上下各设 ``k × 前 N 日区间`` 的触发带：
    向上击穿做多、向下击穿离场。区间用 ``max(HH-LC, HC-LL)``，
    比单纯用 high-low 更能反映真实波动。
    """
    hh = df["high"].rolling(period).max()
    ll = df["low"].rolling(period).min()
    hc = df["close"].rolling(period).max()
    lc = df["close"].rolling(period).min()
    rng = pd.concat([hh - lc, hc - ll], axis=1).max(axis=1).shift(1)
    base = df["open"]
    sig = pd.Series(HOLD, index=df.index, name="volatility_breakout")
    sig[(df["close"] > base + k * rng) & rng.notna()] = BUY
    sig[(df["close"] < base - k * rng) & rng.notna()] = SELL
    return sig


def turtle_score(df: pd.DataFrame, entry: int = 20, adx_gate: float = 20.0) -> pd.Series:
    """海龟突破的**分数版**：位置在通道里的相对高度 × 趋势强度闸门。

    分数版的好处是能进组合投票，也能按强弱定仓——
    刚好碰到上轨和大幅突破，仓位不该一样。
    """
    d = donchian(df, entry)
    width = (d["upper"] - d["lower"]).replace(0, np.nan)
    # 价格在通道中的相对位置：下轨=-1，上轨=+1
    pos = (df["close"] - d["mid"]) / (width / 2)
    gate = (trend_strength(df) / max(adx_gate, 1e-9)).clip(0.0, 1.0)
    return clip_score(pos * gate).rename("turtle_score")


def atr_channel_score(df: pd.DataFrame, period: int = 20, mult: float = 2.0) -> pd.Series:
    """价格偏离 EMA 多少个 ATR，压到 [-1,1]。正=偏强，负=偏弱。"""
    ema = df["close"].ewm(span=period, adjust=False).mean()
    a = atr(df, period).replace(0, np.nan)
    return clip_score((df["close"] - ema) / (a * mult)).rename("atr_channel_score")


def breakout_composite(df: pd.DataFrame, buy_th: float = 0.35,
                       sell_th: float = -0.35) -> pd.Series:
    """把三个突破分数等权合成后转三态，作为「突破」这一大类的代表策略。"""
    s = (turtle_score(df) + atr_channel_score(df)
         + supertrend(df)["trend"].astype("float64")) / 3.0
    return score_to_signal(clip_score(s), buy_th, sell_th).rename("breakout_composite")
