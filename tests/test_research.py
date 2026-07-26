"""研究层单元测试（纯逻辑，不测拉行情的慢路径）。"""

from __future__ import annotations

import pytest


def test_every_default_section_has_handler():
    """默认板块表里的每一项都必须有处理器——写错名字会静默变成 error 板块。

    原来这里断言的是硬编码的板块数（14 / 11 / 4 / 6），加板块必挂一次且
    挂了也不说明哪错了。改成断言两张表的一致性，才真的守住了东西。
    """
    from eq.core.research import _DEFAULT_SECTIONS, _SECTION_HANDLERS
    for market, secs in _DEFAULT_SECTIONS.items():
        assert secs, f"{market} 的默认板块为空"
        for sec in secs:
            assert sec in _SECTION_HANDLERS, f"{market} 的板块 {sec} 没有处理器"


def test_every_market_has_snapshot():
    """行情快照是所有市场的共同底座。"""
    from eq.core.research import _DEFAULT_SECTIONS
    for market, secs in _DEFAULT_SECTIONS.items():
        assert "snapshot" in secs, f"{market} 缺 snapshot"


def test_section_labels():
    """每个处理器都要有中文标签，否则报告里会露出英文键名。"""
    from eq.core.research import _SECTION_HANDLERS, _SECTION_LABELS
    for sec in _SECTION_HANDLERS:
        assert sec in _SECTION_LABELS, f"板块 {sec} 缺中文标签"


def test_research_unknown_section():
    """未知板块应返回 error 而不崩。"""
    from eq.core.research import research
    # 只拉 snapshot 板块避免网络
    report = research("600519.SH", sections=["不存在的板块"])
    assert report["不存在的板块"]["error"]
