"""持仓体检的风险指标 + 输入校验（v0.24 新增）。"""

from __future__ import annotations

import pytest

from eq.core import portfolio as pf


@pytest.fixture
def stub_quotes(monkeypatch):
    """给定 {symbol: 现价} 造快照，避免测试打网络。"""
    def _install(prices: dict[str, float], change_pct: float = 1.0):
        def _fake(symbols, **kwargs):
            return {
                s: ({"symbol": s, "close": prices[s], "change_pct": change_pct,
                     "date": "2026-01-01", "open": prices[s], "high": prices[s],
                     "low": prices[s], "volume": 1e6, "prev_close": prices[s]}
                    if s in prices else None)
                for s in symbols
            }
        monkeypatch.setattr("eq.data.market.get_snapshots", _fake)
    return _install


def test_summary_weights_and_concentration(tmp_db, stub_quotes):
    stub_quotes({"600519.SH": 1000.0, "000001.SZ": 10.0})
    pf.open_position("600519.SH", 100, 900.0)      # 市值 100,000
    pf.open_position("000001.SZ", 1000, 9.0)       # 市值  10,000

    s = pf.summary()
    assert s["total_market_value"] == pytest.approx(110_000)
    assert s["total_cost"] == pytest.approx(99_000)
    assert s["total_unrealized_pnl"] == pytest.approx(11_000)
    assert s["total_unrealized_pct"] == pytest.approx(11_000 / 99_000 * 100)
    by_sym = {p["symbol"]: p for p in s["positions"]}
    assert by_sym["600519.SH"]["weight_pct"] == pytest.approx(100_000 / 110_000 * 100)
    assert s["max_weight_symbol"] == "600519.SH"
    assert s["max_weight_pct"] > 90


def test_summary_risk_at_stop_and_no_stop(tmp_db, stub_quotes):
    stub_quotes({"600519.SH": 1000.0, "000001.SZ": 10.0})
    pf.open_position("600519.SH", 100, 900.0, stop_loss=950.0)   # 触发止损亏 (1000-950)*100
    pf.open_position("000001.SZ", 1000, 9.0)                     # 裸奔

    s = pf.summary()
    assert s["risk_at_stop"] == pytest.approx(5_000)
    assert s["no_stop"] == ["000001.SZ"]


def test_summary_marks_stale_quotes(tmp_db, stub_quotes):
    stub_quotes({"600519.SH": 1000.0})       # 000001.SZ 拉不到
    pf.open_position("600519.SH", 100, 900.0)
    pf.open_position("000001.SZ", 1000, 9.0)

    s = pf.summary()
    assert s["stale"] == ["000001.SZ"]
    by_sym = {p["symbol"]: p for p in s["positions"]}
    assert by_sym["000001.SZ"]["quote_ok"] is False
    # 拉不到行情时用成本价占位，浮盈应为 0 而不是乱数
    assert by_sym["000001.SZ"]["unrealized_pnl"] == pytest.approx(0.0)
    assert by_sym["600519.SH"]["quote_ok"] is True


def test_summary_empty_portfolio(tmp_db):
    s = pf.summary()
    assert s["positions"] == []
    assert s["total_market_value"] == 0
    assert s["max_weight_pct"] == 0
    assert s["max_weight_symbol"] == ""


def test_symbol_variants_map_to_one_position(tmp_db):
    """600519 / 600519.sh / SH600519 应该是同一行持仓，而不是建出三条。"""
    pf.open_position("600519", 100, 1000.0)
    pf.open_position("600519.sh", 100, 1200.0)   # 走加仓分支
    pf.open_position("SH600519", 100, 1400.0)
    assert len(pf.list_open()) == 1
    pos = pf.get_open("600519.SH")
    assert pos["shares"] == 300
    assert pos["cost_price"] == pytest.approx(1200.0)


@pytest.mark.parametrize(("shares", "price"), [(0, 100), (-5, 100), (10, 0), (10, -3)])
def test_open_position_rejects_bad_input(tmp_db, shares, price):
    with pytest.raises(ValueError):
        pf.open_position("600519.SH", shares, price)


def test_trim_rejects_oversell(tmp_db):
    pf.open_position("600519.SH", 100, 1000.0)
    with pytest.raises(ValueError, match="超过持仓"):
        pf.trim("600519.SH", 200, 1100.0)


def test_trim_to_zero_closes_position(tmp_db):
    pf.open_position("600519.SH", 100, 1000.0)
    pf.trim("600519.SH", 100, 1100.0)
    assert pf.get_open("600519.SH") is None
    closed = pf.list_closed()
    assert len(closed) == 1
    assert closed[0]["realized_pnl"] == pytest.approx(10_000)
