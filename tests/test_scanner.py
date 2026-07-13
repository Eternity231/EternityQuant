"""扫描器层单元测试（纯逻辑，不测拉行情的慢路径）。"""

from __future__ import annotations

import pytest


def test_akshare_code_to_eq():
    """A 股代码格式转换：sz302132 → 302132.SZ。"""
    from eq.core.scanner import _akshare_code_to_eq
    assert _akshare_code_to_eq("sz302132") == "302132.SZ"
    assert _akshare_code_to_eq("sh600519") == "600519.SH"
    assert _akshare_code_to_eq("bj920000") == "920000.BJ"
    assert _akshare_code_to_eq("unknown") == "unknown"  # 不识别原样返回


def test_scanners_registered():
    """_SCANNERS 应注册 4 个市场扫描器。"""
    from eq.core.scanner import _SCANNERS
    assert set(_SCANNERS.keys()) == {"A", "HK", "US", "CRYPTO"}


def test_market_labels():
    """_MARKET_LABELS 应覆盖 4 个市场。"""
    from eq.core.scanner import _MARKET_LABELS
    assert _MARKET_LABELS["A"] == "A 股"
    assert _MARKET_LABELS["HK"] == "港股"
    assert _MARKET_LABELS["US"] == "美股"
    assert _MARKET_LABELS["CRYPTO"] == "加密"


def test_scan_unknown_market():
    """未知市场应抛 ValueError。"""
    from eq.core.scanner import scan
    with pytest.raises(ValueError):
        scan("UNKNOWN", top_n=5)
