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


# ---------- file: 股票池来源（v0.42.1） ----------

def test_file_source_parses_one_per_line(tmp_path):
    """冻结股票池：--from A --top N 每次联网重扫，两次可能拿到不同的票，
    跑对照实验时结果彼此不可比且不会报错。file: 让池子可复现。"""
    from eq.cli import _resolve_symbols

    f = tmp_path / "u.txt"
    f.write_text("600519.SH\n000001.SZ\n000858.SZ\n", encoding="utf-8")
    syms, label = _resolve_symbols(f"file:{f}")
    assert syms == ["600519.SH", "000001.SZ", "000858.SZ"]
    assert "3 只" in label


def test_file_source_strips_comments_per_line(tmp_path):
    """注释必须**按行**剥。整体 split 的话，'# 说明文字' 里除了 # 之外的
    每个词都会被当成股票代码——这个 bug 真的发生过。"""
    from eq.cli import _resolve_symbols

    f = tmp_path / "u.txt"
    f.write_text("# EternityQuant ML 实验股票池 冻结自缓存\n"
                 "600519.SH\n"
                 "000001.SZ  # 平安银行\n", encoding="utf-8")
    syms, _ = _resolve_symbols(f"file:{f}")
    assert syms == ["600519.SH", "000001.SZ"]


def test_file_source_accepts_commas_and_blanks(tmp_path):
    from eq.cli import _resolve_symbols

    f = tmp_path / "u.txt"
    f.write_text("600519.SH, 000001.SZ\n\n  \n000858.SZ\n", encoding="utf-8")
    assert _resolve_symbols(f"file:{f}")[0] == ["600519.SH", "000001.SZ", "000858.SZ"]


def test_file_source_dedupes(tmp_path):
    from eq.cli import _resolve_symbols

    f = tmp_path / "u.txt"
    f.write_text("600519.SH\n600519.SH\n000001.SZ\n", encoding="utf-8")
    assert _resolve_symbols(f"file:{f}")[0] == ["600519.SH", "000001.SZ"]


def test_file_source_missing_file_raises():
    from eq.cli import _resolve_symbols

    with pytest.raises(ValueError, match="不存在"):
        _resolve_symbols("file:Z:/不存在/u.txt")


def test_file_source_empty_file_raises(tmp_path):
    from eq.cli import _resolve_symbols

    f = tmp_path / "empty.txt"
    f.write_text("# 只有注释\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="空的"):
        _resolve_symbols(f"file:{f}")
