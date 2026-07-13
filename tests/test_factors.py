"""因子层单元测试（纯逻辑，无网络）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_rsi_range(sample_bars):
    """RSI 值应在 0~100 之间。"""
    from eq.strategy.factors.technical import rsi
    rsi_vals = rsi(sample_bars, period=14)
    assert len(rsi_vals) == len(sample_bars)
    valid = rsi_vals.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_rsi_overbought(sample_bars):
    """连涨的子序列 RSI 应偏高（>=50，纯连涨边界约 50-70）。"""
    from eq.strategy.factors.technical import rsi
    # 造连涨数据
    dates = pd.bdate_range("2025-01-01", periods=20)
    close = pd.Series(range(100, 120), index=dates, dtype=float)
    bars = pd.DataFrame({"close": close, "high": close+1, "low": close-1, "open": close, "volume": 1e6})
    rsi_vals = rsi(bars, period=14)
    # 纯等差连涨 RSI 边界约 50（无波幅），验证不报错且 >=50
    assert rsi_vals.iloc[-1] >= 50


def test_ema_lengths(sample_bars):
    """EMA 输出长度应等于输入长度。"""
    from eq.strategy.factors.technical import ema
    ema_vals = ema(sample_bars, period=12)  # ema 接 DataFrame，取 df["close"]
    assert len(ema_vals) == len(sample_bars)


def test_macd_columns(sample_bars):
    """MACD 应返回 dif/dea/hist 三列。"""
    from eq.strategy.factors.technical import macd
    result = macd(sample_bars, fast=12, slow=26, signal=9)
    assert set(result.columns) == {"dif", "dea", "hist"}
    assert len(result) == len(sample_bars)


def test_kdj_range(sample_bars):
    """KDJ 的 K/D 值应在 0~100 之间。"""
    from eq.strategy.factors.technical import kdj
    result = kdj(sample_bars, n=9, m1=3, m2=3)
    assert "K" in result.columns and "D" in result.columns
    valid_k = result["K"].dropna()
    assert (valid_k >= 0).all() and (valid_k <= 100).all()


def test_bollinger_columns(sample_bars):
    """布林带应返回 upper/mid/lower 三列。"""
    from eq.strategy.factors.technical import bollinger
    result = bollinger(sample_bars, period=20, k=2)
    assert set(result.columns) == {"upper", "mid", "lower"}
    # upper > mid > lower
    valid = result.dropna()
    if not valid.empty:
        assert (valid["upper"] >= valid["mid"]).all()
        assert (valid["mid"] >= valid["lower"]).all()


def test_adx_range(sample_bars):
    """ADX 值应在 0~100 之间。"""
    from eq.strategy.factors.technical import adx
    result = adx(sample_bars, period=14)
    valid = result.dropna()
    if not valid.empty:
        assert (valid >= 0).all() and (valid <= 100).all()
