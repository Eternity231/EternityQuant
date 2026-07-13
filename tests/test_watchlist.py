"""自选股层单元测试（用 tmp_db fixture）。"""

from __future__ import annotations

import pytest


def test_add_and_list(tmp_db):
    from eq.core import watchlist as wl
    wl.add("600519.SH", reason="白酒龙头", tags="白酒,龙头")
    rows = wl.list_all()
    assert len(rows) == 1
    assert rows[0]["symbol"] == "600519.SH"
    assert rows[0]["reason"] == "白酒龙头"


def test_find(tmp_db):
    from eq.core import watchlist as wl
    wl.add("600519.SH")
    found = wl.find("600519.SH")
    assert found is not None
    assert found["symbol"] == "600519.SH"
    assert wl.find("000001.SZ") is None


def test_remove(tmp_db):
    from eq.core import watchlist as wl
    wl.add("600519.SH")
    assert wl.remove("600519.SH") is True
    assert wl.find("600519.SH") is None
    assert wl.remove("600519.SH") is False  # 已删


def test_update_name(tmp_db):
    from eq.core import watchlist as wl
    wl.add("600519.SH")
    wl.update_name("600519.SH", "贵州茅台")
    found = wl.find("600519.SH")
    assert found["name"] == "贵州茅台"
