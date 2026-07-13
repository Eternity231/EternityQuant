"""研究层单元测试（纯逻辑，不测拉行情的慢路径）。"""

from __future__ import annotations

import pytest


def test_section_handlers_count():
    """_SECTION_HANDLERS 应注册 14 个板块处理器。"""
    from eq.core.research import _SECTION_HANDLERS
    assert len(_SECTION_HANDLERS) == 14


def test_default_sections_by_market():
    """_DEFAULT_SECTIONS 应按市场定义板块。"""
    from eq.core.research import _DEFAULT_SECTIONS
    assert len(_DEFAULT_SECTIONS["A"]) == 11
    assert len(_DEFAULT_SECTIONS["HK"]) == 4
    assert len(_DEFAULT_SECTIONS["US"]) == 6
    assert len(_DEFAULT_SECTIONS["CRYPTO"]) == 1
    assert "snapshot" in _DEFAULT_SECTIONS["A"]
    assert "snapshot" in _DEFAULT_SECTIONS["HK"]


def test_section_labels():
    """_SECTION_LABELS 应覆盖全部 14 板块的中文名。"""
    from eq.core.research import _SECTION_LABELS, _SECTION_HANDLERS
    for sec in _SECTION_HANDLERS:
        assert sec in _SECTION_LABELS, f"板块 {sec} 缺中文标签"


def test_research_unknown_section():
    """未知板块应返回 error 而不崩。"""
    from eq.core.research import research
    # 只拉 snapshot 板块避免网络
    report = research("600519.SH", sections=["不存在的板块"])
    assert report["不存在的板块"]["error"]
