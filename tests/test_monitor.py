"""监控规则层单元测试（用 tmp_db fixture）。"""

from __future__ import annotations

import pytest


def test_validate_params():
    """_validate_params 应校验必填字段。"""
    from eq.core.monitor import _validate_params, RULE_TYPES
    # 正确参数
    _validate_params("price_cross", {"level": 1700, "direction": "up"})
    _validate_params("price_pct", {"threshold": 5.0})
    # 缺字段应抛错
    with pytest.raises(ValueError):
        _validate_params("price_cross", {"level": 1700})  # 缺 direction
    with pytest.raises(ValueError):
        _validate_params("price_pct", {})  # 缺 threshold


def test_rule_types_count():
    """RULE_TYPES 应有 11 种（含 stop_loss/take_profit）。"""
    from eq.core.monitor import RULE_TYPES
    assert len(RULE_TYPES) == 11
    assert "indicator" in RULE_TYPES
    assert "news" in RULE_TYPES
    assert "flow" in RULE_TYPES
    assert "stop_loss" in RULE_TYPES


def test_add_and_list_rule(tmp_db):
    from eq.core import monitor as mon
    mon.add_rule("600519.SH", "price_cross", {"level": 1700, "direction": "up"}, channels=["desktop"])
    rules = mon.list_rules()
    assert len(rules) == 1
    assert rules[0]["symbol"] == "600519.SH"
    assert rules[0]["type"] == "price_cross"  # 列名是 type 不是 rule_type
    assert rules[0]["enabled"] == 1  # 默认启用


def test_remove_rule(tmp_db):
    from eq.core import monitor as mon
    rid = mon.add_rule("600519.SH", "price_pct", {"threshold": 5.0})
    assert mon.remove_rule(rid) is True
    assert mon.list_rules() == []


def test_set_enabled(tmp_db):
    from eq.core import monitor as mon
    rid = mon.add_rule("600519.SH", "price_pct", {"threshold": 5.0})
    assert mon.set_enabled(rid, False) is True
    rules = mon.list_rules()
    assert rules[0]["enabled"] == 0
    assert mon.set_enabled(999, True) is False  # 不存在


def test_handlers_registered():
    """_HANDLERS 应注册全部 11 种规则类型处理器。"""
    from eq.core.monitor import _HANDLERS
    assert len(_HANDLERS) == 11
    for t in ["price_cross", "price_pct", "indicator", "volume_spike",
              "limit_up", "limit_down", "news", "event", "flow",
              "stop_loss", "take_profit"]:
        assert t in _HANDLERS
