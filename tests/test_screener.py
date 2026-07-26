"""技术选股器（v0.24 新增）。用 monkeypatch 换掉取数，纯逻辑无网络。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eq.core import screener as sc


def _bars(close: np.ndarray) -> pd.DataFrame:
    n = len(close)
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": np.full(n, 1e6),
    }, index=idx)


@pytest.fixture
def fake_bars(monkeypatch):
    """把 screener 的取数换成内存数据：涨的、跌的、放量的各一只。"""
    n = 120
    up = _bars(np.linspace(50, 100, n))                       # 单边上涨
    down = _bars(np.linspace(100, 50, n))                     # 单边下跌
    spike = _bars(np.full(n, 60.0) + np.sin(np.arange(n)))    # 震荡
    spike.loc[spike.index[-1], "volume"] = 1e7                # 末根量能放大 10x

    data = {"600000.SH": up, "600001.SH": down, "600002.SH": spike}

    def _fake(symbol, days=30, **kwargs):
        if symbol not in data:
            raise ValueError(f"no data: {symbol}")
        return data[symbol].tail(days)

    monkeypatch.setattr(sc, "get_recent_bars", _fake)
    return list(data)


def test_conditions_registry_complete():
    """每个条件都要能被调用且有中文标签。"""
    assert len(sc.CONDITIONS) >= 14
    for name, (fn, label) in sc.CONDITIONS.items():
        assert callable(fn), name
        assert label, name


def test_screen_rejects_unknown_condition(fake_bars):
    with pytest.raises(ValueError, match="未知筛选条件"):
        sc.screen(fake_bars, ["不存在的条件"])


def test_screen_rejects_empty_conditions(fake_bars):
    with pytest.raises(ValueError):
        sc.screen(fake_bars, [])


def test_screen_rejects_bad_mode(fake_bars):
    with pytest.raises(ValueError, match="mode"):
        sc.screen(fake_bars, ["above_ma"], mode="whatever")


def test_above_ma_picks_uptrend_only(fake_bars):
    hits = sc.screen(fake_bars, ["above_ma"], workers=2)
    syms = {h["symbol"] for h in hits}
    assert "600000.SH" in syms      # 单边上涨必然站上 MA20
    assert "600001.SH" not in syms  # 单边下跌必然在 MA20 之下


def test_below_ma_picks_downtrend_only(fake_bars):
    syms = {h["symbol"] for h in sc.screen(fake_bars, ["below_ma"], workers=2)}
    assert "600001.SH" in syms
    assert "600000.SH" not in syms


def test_volume_spike(fake_bars):
    syms = {h["symbol"] for h in sc.screen(fake_bars, ["volume_spike"], workers=2)}
    assert syms == {"600002.SH"}


def test_mode_all_vs_any(fake_bars):
    all_hits = sc.screen(fake_bars, ["above_ma", "volume_spike"], mode="all", workers=2)
    any_hits = sc.screen(fake_bars, ["above_ma", "volume_spike"], mode="any", workers=2)
    assert len(all_hits) == 0          # 没有一只同时上涨 + 放量
    assert len(any_hits) >= 2          # 各自都有命中
    assert all(h["score"] >= 1 for h in any_hits)


def test_results_sorted_by_score(fake_bars):
    hits = sc.screen(fake_bars, ["above_ma", "near_high", "breakout"], mode="any", workers=2)
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_missing_symbol_is_skipped_not_raised(fake_bars):
    hits = sc.screen([*fake_bars, "999999.SH"], ["above_ma"], workers=2)
    assert "999999.SH" not in {h["symbol"] for h in hits}


def test_format_screen_empty_and_nonempty(fake_bars):
    assert "无标的命中" in sc.format_screen([], ["above_ma"])
    hits = sc.screen(fake_bars, ["above_ma"], workers=2)
    out = sc.format_screen(hits, ["above_ma"])
    assert "600000.SH" in out
