"""监控规则冷却期 + 信号落库（v0.24 新增）。"""

from __future__ import annotations

import datetime as dt

import pytest

from eq.core import monitor as mon


def test_add_rule_with_cooldown_stores_param(tmp_db):
    rid = mon.add_rule("600519.SH", "price_pct", {"threshold": 5.0}, cooldown_minutes=30)
    rule = next(r for r in mon.list_rules() if r["id"] == rid)
    assert rule["params"]["cooldown_minutes"] == 30


def test_add_rule_without_cooldown_omits_param(tmp_db):
    rid = mon.add_rule("600519.SH", "price_pct", {"threshold": 5.0})
    rule = next(r for r in mon.list_rules() if r["id"] == rid)
    assert "cooldown_minutes" not in rule["params"]


def test_add_rule_normalizes_symbol(tmp_db):
    rid = mon.add_rule("600519.sh", "price_pct", {"threshold": 5.0})
    rule = next(r for r in mon.list_rules() if r["id"] == rid)
    assert rule["symbol"] == "600519.SH"


def test_add_rule_does_not_mutate_caller_params(tmp_db):
    params = {"threshold": 5.0}
    mon.add_rule("600519.SH", "price_pct", params, cooldown_minutes=30)
    assert params == {"threshold": 5.0}, "不应就地改调用方传进来的 dict"


def test_in_cooldown_logic():
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    recent = (now - dt.timedelta(minutes=5)).isoformat(sep=" ")
    old = (now - dt.timedelta(minutes=120)).isoformat(sep=" ")

    # 冷却期内
    assert mon._in_cooldown({"params": {"cooldown_minutes": 60}, "last_fired_at": recent}) is True
    # 冷却期已过
    assert mon._in_cooldown({"params": {"cooldown_minutes": 60}, "last_fired_at": old}) is False
    # 没设冷却
    assert mon._in_cooldown({"params": {}, "last_fired_at": recent}) is False
    # 从没触发过
    assert mon._in_cooldown({"params": {"cooldown_minutes": 60}, "last_fired_at": None}) is False
    # 时间戳解析不了也不该崩
    assert mon._in_cooldown({"params": {"cooldown_minutes": 60}, "last_fired_at": "垃圾"}) is False


def test_set_cooldown(tmp_db):
    rid = mon.add_rule("600519.SH", "price_pct", {"threshold": 5.0})
    assert mon.set_cooldown(rid, 45) is True
    rule = next(r for r in mon.list_rules() if r["id"] == rid)
    assert rule["params"]["cooldown_minutes"] == 45
    assert rule["params"]["threshold"] == 5.0     # 原参数不能丢

    assert mon.set_cooldown(rid, 0) is True
    rule = next(r for r in mon.list_rules() if r["id"] == rid)
    assert "cooldown_minutes" not in rule["params"]

    assert mon.set_cooldown(9999, 10) is False


def test_evaluate_skips_when_in_cooldown(tmp_db, monkeypatch):
    """冷却期内应直接跳过，连 handler 都不该调（省掉一次网络往返）。"""
    calls = []
    monkeypatch.setitem(mon._HANDLERS, "price_pct",
                        lambda rule: (calls.append(1), (True, "t", "b"))[1])

    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    rule = {
        "id": 1, "symbol": "600519.SH", "type": "price_pct", "channels": [],
        "params": {"threshold": 1.0, "cooldown_minutes": 60},
        "last_fired_at": (now - dt.timedelta(minutes=1)).isoformat(sep=" "),
    }
    assert mon._evaluate(rule) is False
    assert not calls, "冷却期内不该调用 handler"


def test_evaluate_fires_and_records_signal(tmp_db, monkeypatch):
    monkeypatch.setitem(mon._HANDLERS, "price_pct", lambda rule: (True, "涨跌幅异动", "正文"))
    rid = mon.add_rule("600519.SH", "price_pct", {"threshold": 1.0}, channels=[])
    rule = next(r for r in mon.list_rules() if r["id"] == rid)

    assert mon._evaluate(rule) is True
    sigs = mon.recent_signals()
    assert len(sigs) == 1
    assert sigs[0]["symbol"] == "600519.SH"
    assert sigs[0]["signal_type"] == "price_pct"
    assert sigs[0]["context"]["title"] == "涨跌幅异动"
    assert sigs[0]["context"]["rule_id"] == rid


def test_recent_signals_filter_by_symbol(tmp_db, monkeypatch):
    monkeypatch.setitem(mon._HANDLERS, "price_pct", lambda rule: (True, "t", "b"))
    for sym in ("600519.SH", "000001.SZ"):
        rid = mon.add_rule(sym, "price_pct", {"threshold": 1.0}, channels=[])
        mon._evaluate(next(r for r in mon.list_rules() if r["id"] == rid))
    assert len(mon.recent_signals()) == 2
    assert len(mon.recent_signals(symbol="600519.SH")) == 1
    assert len(mon.recent_signals(symbol="600519.sh")) == 1  # 符号规整


@pytest.mark.parametrize(("symbol", "band"), [
    ("600519.SH", 0.10),   # 主板
    ("000001.SZ", 0.10),   # 深主板
    ("300750.SZ", 0.20),   # 创业板
    ("688111.SH", 0.20),   # 科创板
    ("920000.BJ", 0.30),   # 北交所
])
def test_limit_band_by_board(symbol, band):
    """此前一律按 ±10% 算，创业板/科创板/北交所的涨跌停规则永远差 10~20 个点触发不了。"""
    assert mon.limit_band(symbol) == band
