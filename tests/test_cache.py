"""行情缓存层（v0.24 新增）。bar_cache 表此前建了但从没人读写。"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from eq.data import cache


def _make_bars(end: dt.date, n: int = 20) -> pd.DataFrame:
    idx = pd.bdate_range(end=pd.Timestamp(end), periods=n)
    close = np.linspace(10, 12, n)
    return pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": np.full(n, 1e6),
    }, index=idx)


@pytest.fixture
def bars():
    """固定区间的 20 根日线，测缓存读写本身。"""
    return _make_bars(dt.date(2025, 1, 28))


@pytest.fixture
def recent_bars():
    """截止今天的 60 根日线——get_recent_bars 查的是 [今天-2N, 今天] 窗口，
    用陈年日期的数据落不进查询区间。"""
    return _make_bars(dt.date.today(), n=60)


def test_save_and_load_roundtrip(tmp_db, bars):
    n = cache.save_bars("600519.SH", bars)
    assert n == 20
    got = cache.load_bars("600519.SH", dt.date(2025, 1, 1), dt.date(2025, 2, 1))
    assert len(got) == 20
    assert list(got.columns) == ["open", "high", "low", "close", "volume"]
    assert got["close"].iloc[-1] == pytest.approx(12.0)


def test_save_is_idempotent_upsert(tmp_db, bars):
    cache.save_bars("600519.SH", bars)
    cache.save_bars("600519.SH", bars)          # 再写一次不该翻倍
    assert len(cache.load_bars("600519.SH", dt.date(2025, 1, 1), dt.date(2025, 2, 1))) == 20


def test_save_updates_existing_rows(tmp_db, bars):
    cache.save_bars("600519.SH", bars)
    bumped = bars.copy()
    bumped["close"] = bumped["close"] + 100
    cache.save_bars("600519.SH", bumped)
    got = cache.load_bars("600519.SH", dt.date(2025, 1, 1), dt.date(2025, 2, 1))
    assert got["close"].iloc[-1] == pytest.approx(112.0)


def test_load_empty_for_unknown_symbol(tmp_db):
    assert cache.load_bars("000000.SZ", dt.date(2025, 1, 1), dt.date(2025, 2, 1)).empty


def test_save_ignores_bad_input(tmp_db):
    assert cache.save_bars("X.SH", pd.DataFrame()) == 0
    assert cache.save_bars("X.SH", None) == 0
    assert cache.save_bars("X.SH", pd.DataFrame({"foo": [1]})) == 0   # 缺 OHLCV 列


def test_is_fresh_respects_ttl(tmp_db, bars):
    cache.save_bars("600519.SH", bars)
    start, end = dt.date(2025, 1, 1), dt.date(2025, 1, 30)
    assert cache.is_fresh("600519.SH", start, end, ttl_seconds=3600) is True
    assert cache.is_fresh("600519.SH", start, end, ttl_seconds=0) is False


def test_is_fresh_false_when_range_not_covered(tmp_db, bars):
    cache.save_bars("600519.SH", bars)
    # 起点早于缓存起点 → 不新鲜
    assert cache.is_fresh("600519.SH", dt.date(2020, 1, 1), dt.date(2025, 1, 30)) is False
    # 终点远晚于缓存终点 → 不新鲜
    assert cache.is_fresh("600519.SH", dt.date(2025, 1, 1), dt.date(2025, 6, 1)) is False


def test_is_fresh_false_for_unknown(tmp_db):
    assert cache.is_fresh("999999.SH", dt.date(2025, 1, 1), dt.date(2025, 1, 30)) is False


def test_stats_and_clear(tmp_db, bars):
    cache.save_bars("600519.SH", bars)
    cache.save_bars("000001.SZ", bars)
    s = cache.stats()
    assert s["symbols"] == 2
    assert s["rows"] == 40
    assert len(s["per_symbol"]) == 2

    assert cache.clear("600519.SH") == 20
    assert cache.stats()["symbols"] == 1
    cache.clear()
    assert cache.stats()["rows"] == 0


def test_get_recent_bars_uses_cache(tmp_db, recent_bars, monkeypatch):
    """缓存新鲜时不该再打网络。"""
    from eq.data import market

    cache.save_bars("600519.SH", recent_bars)

    def _boom(*a, **k):
        raise AssertionError("缓存命中时不应该走网络")

    monkeypatch.setattr(market, "_fetch_baostock_a", _boom)
    monkeypatch.setattr(market, "_fetch_akshare_fallback", _boom)

    got = market.get_recent_bars("600519.SH", days=5)
    assert len(got) == 5
    assert got["close"].iloc[-1] == pytest.approx(12.0)


def test_get_recent_bars_falls_back_to_stale_cache(tmp_db, recent_bars, monkeypatch):
    """网络全挂时应退化用过期缓存，而不是直接抛错。"""
    from eq.data import market

    cache.save_bars("600519.SH", recent_bars)
    monkeypatch.setattr(cache, "is_fresh", lambda *a, **k: False)  # 强制视作过期
    monkeypatch.setattr(market, "_fetch_baostock_a",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("主源挂了")))
    monkeypatch.setattr(market, "_fetch_akshare_fallback",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("兜底也挂了")))

    got = market.get_recent_bars("600519.SH", days=10)
    assert len(got) == 10


def test_get_recent_bars_no_cache_flag_skips_cache(tmp_db, recent_bars, monkeypatch):
    """use_cache=False 时即使缓存新鲜也必须走网络。

    v0.26 起取数走 eq.data.sources 注册表，所以要在注册表层拦截，
    不能再 patch market._fetch_baostock_a（那条路已经不是主路径了）。
    """
    from eq.data import market, sources as sr

    cache.save_bars("600519.SH", recent_bars)
    called = []
    monkeypatch.setattr(sr, "fetch_bars",
                        lambda *a, **k: (called.append(1), (recent_bars, "fake"))[1])

    market.get_recent_bars("600519.SH", days=5, use_cache=False)
    assert called, "use_cache=False 应该走网络"
