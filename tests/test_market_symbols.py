"""符号规整与市场识别（v0.24 新增，纯逻辑无网络）。"""

from __future__ import annotations

import pytest

from eq.data.market import bare_code, detect_market, normalize_symbol


@pytest.mark.parametrize(("raw", "expected"), [
    # 大小写 / 空白
    ("600519.sh", "600519.SH"),
    ("  600519.SH  ", "600519.SH"),
    ('"600519.SH"', "600519.SH"),
    # 裸 A 股 6 位按板块补后缀
    ("600519", "600519.SH"),
    ("000001", "000001.SZ"),
    ("300750", "300750.SZ"),
    ("920000", "920000.BJ"),
    # qlib 风格前缀
    ("SH600519", "600519.SH"),
    ("sz000001", "000001.SZ"),
    # 港股补零
    ("700", "00700.HK"),
    ("0700.HK", "00700.HK"),
    ("09988.hk", "09988.HK"),
    ("5", "00005.HK"),
    # 美股
    ("AAPL", "AAPL.US"),
    ("aapl.us", "AAPL.US"),
    # 加密原样
    ("BTC-USDT", "BTC-USDT"),
])
def test_normalize_symbol(raw, expected):
    assert normalize_symbol(raw) == expected


@pytest.mark.parametrize(("raw", "market"), [
    ("600519.sh", "A"), ("000001", "A"), ("SH600519", "A"),
    ("00700.HK", "HK"), ("700", "HK"),
    ("AAPL", "US"), ("aapl.us", "US"),
    ("BTC-USDT", "CRYPTO"),
])
def test_detect_market_accepts_loose_input(raw, market):
    """此前 detect_market 只认严格大写全后缀格式，其余一律 ValueError。"""
    assert detect_market(raw) == market


def test_detect_market_rejects_garbage():
    with pytest.raises(ValueError):
        detect_market("这不是代码!!")


def test_bare_code():
    assert bare_code("600519.SH") == "600519"
    assert bare_code("600519") == "600519"
    assert bare_code("00700.HK") == "00700"


def test_yfinance_hk_symbol_is_four_digits():
    """yfinance 港股用 4 位零填充（0700.HK），传 5 位查无此票。"""
    from eq.data.market import yfinance_symbol

    assert yfinance_symbol("00700.HK", "HK") == "0700.HK"
    assert yfinance_symbol("09988.HK", "HK") == "9988.HK"


def test_yfinance_crypto_maps_usdt_to_usd():
    from eq.data.market import yfinance_symbol

    assert yfinance_symbol("BTC-USDT", "CRYPTO") == "BTC-USD"
