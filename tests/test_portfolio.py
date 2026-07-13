"""持仓层单元测试（用 tmp_db fixture）。"""

from __future__ import annotations

import pytest


def test_open_and_list(tmp_db):
    """建仓后 list_open 应能读到。"""
    from eq.core import portfolio as pf
    pf.open_position("600519.SH", 100, 1680.0)
    rows = pf.list_open()
    assert len(rows) == 1
    assert rows[0]["symbol"] == "600519.SH"
    assert rows[0]["shares"] == 100
    assert rows[0]["cost_price"] == 1680.0


def test_add_recompute_cost(tmp_db):
    """加仓后成本价应加权平均。"""
    from eq.core import portfolio as pf
    pf.open_position("600519.SH", 100, 1680.0)
    pf.add("600519.SH", 100, 1800.0)
    pos = pf.get_open("600519.SH")
    assert pos["shares"] == 200
    # 加权成本 = (100*1680 + 100*1800) / 200 = 1740
    assert abs(pos["cost_price"] - 1740.0) < 0.01


def test_trim_reduce_shares(tmp_db):
    """减仓后 shares 应减少。"""
    from eq.core import portfolio as pf
    pf.open_position("600519.SH", 100, 1680.0)
    pf.trim("600519.SH", 30, 1700.0)
    pos = pf.get_open("600519.SH")
    assert pos["shares"] == 70


def test_set_stops(tmp_db):
    """止损止盈价应能设置。"""
    from eq.core import portfolio as pf
    pf.open_position("600519.SH", 100, 1680.0)
    assert pf.set_stops("600519.SH", stop_loss=1600.0, take_profit=1800.0)
    pos = pf.get_open("600519.SH")
    assert pos["stop_loss"] == 1600.0
    assert pos["take_profit"] == 1800.0


def test_trade_history(tmp_db):
    """交易历史应记录所有动作。"""
    from eq.core import portfolio as pf
    pf.open_position("600519.SH", 100, 1680.0)
    pf.add("600519.SH", 50, 1700.0)
    pf.trim("600519.SH", 30, 1750.0)
    hist = pf.trade_history("600519.SH")
    assert len(hist) >= 3  # open + add + trim


def test_get_open_nonexistent(tmp_db):
    """不存在的标的 get_open 应返回 None。"""
    from eq.core import portfolio as pf
    assert pf.get_open("000001.SZ") is None
