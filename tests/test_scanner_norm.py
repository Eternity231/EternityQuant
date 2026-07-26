"""扫描器列规整与格式化（v0.24 修复项的回归测试，纯逻辑无网络）。"""

from __future__ import annotations

import pandas as pd
import pytest

from eq.core import scanner as sc

_COL_MAP = {
    "代码": "symbol", "名称": "name", "最新价": "close",
    "涨跌幅": "change_pct", "成交量": "volume", "成交额": "amount",
}


def _raw(n=5, with_volume=True):
    d = {
        "代码": [f"sh60000{i}" for i in range(n)],
        "名称": [f"股{i}" for i in range(n)],
        "最新价": [10.0 + i for i in range(n)],
        "涨跌幅": [float(n - i) for i in range(n)],
    }
    if with_volume:
        d["成交量"] = [1e6 * (i + 1) for i in range(n)]
        d["成交额"] = [1e7 * (i + 1) for i in range(n)]
    return pd.DataFrame(d)


def test_norm_cols_sorts_and_truncates():
    out = sc._norm_cols(_raw(10), _COL_MAP, "change_pct", 3)
    assert len(out) == 3
    assert out["change_pct"].is_monotonic_decreasing


def test_norm_cols_tolerates_missing_columns():
    """上游 akshare 少给列时不该 KeyError 把整个命令打挂。"""
    out = sc._norm_cols(_raw(5, with_volume=False), _COL_MAP, "change_pct", 3)
    assert len(out) == 3
    assert "volume" not in out.columns


def test_norm_cols_falls_back_when_sort_key_absent():
    """排序键不在列里时，此前既不排序也不截断——整表原样返回。"""
    out = sc._norm_cols(_raw(20, with_volume=False), _COL_MAP, "volume", 5)
    assert len(out) == 5, "必须截断到 top_n"
    assert out["change_pct"].is_monotonic_decreasing, "应退回按涨跌幅排序"


def test_norm_cols_raises_without_symbol_column():
    with pytest.raises(ValueError, match="代码列"):
        sc._norm_cols(pd.DataFrame({"foo": [1]}), _COL_MAP, "change_pct", 5)


def test_norm_cols_drops_unparseable_rows():
    raw = _raw(3)
    raw["最新价"] = raw["最新价"].astype(object)
    raw.loc[1, "最新价"] = "停牌"
    out = sc._norm_cols(raw, _COL_MAP, "change_pct", 10)
    assert len(out) == 2


def test_akshare_code_conversion():
    assert sc._akshare_code_to_eq("sz302132") == "302132.SZ"
    assert sc._akshare_code_to_eq("sh600519") == "600519.SH"
    assert sc._akshare_code_to_eq("bj920000") == "920000.BJ"
    assert sc._akshare_code_to_eq("weird") == "weird"


def test_us_symbol_handles_bare_code(monkeypatch):
    """东财美股代码通常是 105.NVDA，偶尔回裸 NVDA。
    此前 split('.')[1] 对裸代码得到 NaN，符号变成 nan.US。"""
    raw = pd.DataFrame({
        "代码": ["105.NVDA", "AAPL", "106.BRK.B"],
        "名称": ["英伟达", "苹果", "伯克希尔"],
        "最新价": [100.0, 200.0, 300.0],
        "涨跌幅": [1.0, 2.0, 3.0],
        "开盘价": [1, 2, 3], "最高价": [1, 2, 3],
        "最低价": [1, 2, 3], "昨收价": [1, 2, 3],
    })
    monkeypatch.setattr("akshare.stock_us_famous_spot_em", lambda: raw, raising=False)
    out = sc.scan_us(top_n=10)
    syms = set(out["symbol"])
    assert "NVDA.US" in syms
    assert "AAPL.US" in syms
    assert not any("nan" in s.lower() for s in syms)


def test_format_scan_no_color_when_disabled():
    df = pd.DataFrame({
        "symbol": ["600519.SH"], "name": ["贵州茅台"], "close": [1680.0],
        "change_pct": [2.5], "volume": [1e6], "amount": [1e9],
    })
    plain = sc.format_scan(df, "change_pct", "A", color=False)
    assert "\033[" not in plain, "关色时不该有 ANSI 转义序列"
    assert "600519.SH" in plain
    colored = sc.format_scan(df, "change_pct", "A", color=True)
    assert "\033[" in colored


def test_format_scan_handles_missing_optional_columns():
    """没有 volume/amount 列时也不该崩。"""
    df = pd.DataFrame({"symbol": ["AAPL.US"], "name": ["苹果"],
                       "close": [200.0], "change_pct": [1.5]})
    out = sc.format_scan(df, "change_pct", "US", color=False)
    assert "AAPL.US" in out


def test_use_color_respects_no_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert sc.use_color() is False
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("EQ_FORCE_COLOR", "1")
    assert sc.use_color() is True
