"""深度研究去 MCP 化（v0.34）。

在 v0.33 之前，港美的 profile/financial/news/research/sec_filings/options
六个板块只返回一句"建议调 vibe-trading MCP get_xxx"——港股 4 个板块里 3 个、
美股 6 个里 5 个是占位符，命令行单独跑等于没内容。这套用例钉死两件事：

1. 代码里**不允许**再出现 MCP/vibe-trading 的引用（含 hint 字段）
2. 港美各板块用假 yfinance 喂数据后，能真的产出结构化结果

用假 yfinance 而不是打真网络：单测不该依赖外网和第三方接口的当日可用性。
"""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from eq.core import research


# ---------- 1. 不再有任何 MCP 痕迹 ----------

def test_no_mcp_references_in_source():
    import inspect
    src = inspect.getsource(research)
    for bad in ("vibe-trading", "vibe_trading", "MCP", "mcp__"):
        assert bad not in src, f"research.py 里仍残留 {bad}"


def test_no_hint_field_anywhere():
    """hint 字段整体换成 note；渲染层也只认 note。"""
    import inspect
    assert '"hint"' not in inspect.getsource(research)


def test_mcp_config_file_removed():
    from pathlib import Path
    root = Path(research.__file__).resolve().parents[2]
    assert not (root / ".mcp.json").exists(), ".mcp.json 应已删除"


# ---------- 2. 假 yfinance ----------

class _FakeChain:
    def __init__(self):
        self.calls = pd.DataFrame({
            "strike": [90.0, 100.0, 110.0],
            "openInterest": [10.0, 200.0, 30.0],
            "impliedVolatility": [0.55, 0.42, 0.61],
        })
        self.puts = pd.DataFrame({
            "strike": [90.0, 100.0, 110.0],
            "openInterest": [50.0, 100.0, 30.0],
            "impliedVolatility": [0.6, 0.45, 0.5],
        })


class _FakeTicker:
    def __init__(self, sym):
        self.ticker = sym

    info = {
        "longName": "Fake Corp", "sector": "Technology", "industry": "Software",
        "country": "United States", "fullTimeEmployees": 1234,
        "marketCap": 1e11, "trailingPE": 25.5, "profitMargins": 0.21,
        "returnOnEquity": 0.33, "longBusinessSummary": "做假数据的公司。" * 40,
    }

    @property
    def income_stmt(self):
        return pd.DataFrame(
            {pd.Timestamp("2025-12-31"): [1e10, 4e9, 2e9, 1.5e9],
             pd.Timestamp("2024-12-31"): [9e9, 3.5e9, 1.8e9, 1.2e9]},
            index=["Total Revenue", "Gross Profit", "Operating Income", "Net Income"],
        )

    @property
    def news(self):
        return [{"content": {"title": "假新闻标题一", "pubDate": "2026-07-25T08:00:00Z",
                             "provider": {"displayName": "FakeWire"}}},
                {"content": {"title": "假新闻标题二", "pubDate": "2026-07-24T08:00:00Z",
                             "provider": {"displayName": "FakeWire"}}}]

    @property
    def analyst_price_targets(self):
        return {"current": 100.0, "mean": 125.0, "high": 160.0, "low": 90.0}

    @property
    def recommendations(self):
        return pd.DataFrame({"period": ["0m"], "strongBuy": [5], "buy": [10],
                             "hold": [3], "sell": [1], "strongSell": [0]})

    @property
    def major_holders(self):
        return pd.DataFrame({"Value": ["0.12%", "78.5%"]},
                            index=["insidersPercentHeld", "institutionsPercentHeld"])

    @property
    def institutional_holders(self):
        return pd.DataFrame({"Holder": ["Big Fund", "Other Fund"],
                             "Shares": [1000000, 500000], "pctHeld": [0.05, 0.02]})

    @property
    def sec_filings(self):
        return pd.DataFrame({"date": ["2026-05-01"], "type": ["10-Q"],
                             "title": ["Quarterly report"]})

    options = ("2026-08-21", "2026-09-18")

    def option_chain(self, exp):
        return _FakeChain()


@pytest.fixture
def fake_yf(monkeypatch):
    mod = types.ModuleType("yfinance")
    mod.Ticker = _FakeTicker
    monkeypatch.setitem(sys.modules, "yfinance", mod)
    return mod


# ---------- 3. 港美板块真的出内容 ----------

def test_profile_us(fake_yf):
    out = research._h_profile("AAPL.US", "US")
    p = out["profile"]
    assert p["名称"] == "Fake Corp" and p["行业"] == "Software"
    assert len(p["主营"]) <= 401, "长简介应截断，否则报告刷屏"


def test_financial_us(fake_yf):
    out = research._h_financial("AAPL.US", "US")
    assert out["metrics"]["市盈率TTM"] == 25.5
    assert out["metrics"]["ROE"] == "33.00%", "比例字段要转成百分比"
    assert "2025-12-31" in out["income_stmt"]
    assert out["income_stmt"]["2025-12-31"]["Total Revenue"] == 1e10


def test_dividend_yield_not_multiplied(fake_yf, monkeypatch):
    """yfinance 的 dividendYield 本来就是百分数，不能再 ×100。

    实测 AAPL：info 返回 0.32，而 dividendRate 1.08 / price 333.02 = 0.324%。
    统一 ×100 会打出 32%，把股息率放大 100 倍——同一个 info 里
    profitMargins/ROE 却是小数，这是 yfinance 自己的不一致。
    """
    info = dict(_FakeTicker.info)
    info["dividendYield"] = 0.32
    monkeypatch.setattr(_FakeTicker, "info", info)
    m = research._h_financial("AAPL.US", "US")["metrics"]
    assert m["股息率"] == "0.32%"
    assert m["净利率"] == "21.00%", "小数型比例仍要 ×100"


def test_news_hk(fake_yf):
    out = research._h_news("00700.HK", "HK")
    assert len(out["headlines"]) == 2
    assert out["headlines"][0]["title"] == "假新闻标题一"
    assert out["headlines"][0]["publisher"] == "FakeWire"


def test_research_us_gives_targets(fake_yf):
    out = research._h_research("AAPL.US", "US")
    assert out["targets"]["mean"] == 125.0
    assert out["ratings"][0]["strongBuy"] == 5


def test_holders_us(fake_yf):
    out = research._h_holders("AAPL.US", "US")
    assert out["major"]["institutionsPercentHeld"] == "78.5%"
    assert out["institutional"][0]["Holder"] == "Big Fund"


def test_sec_filings_us(fake_yf):
    out = research._h_sec_filings("AAPL.US", "US")
    assert out["filings"][0]["type"] == "10-Q"


def test_options_us(fake_yf, monkeypatch):
    monkeypatch.setattr(research, "get_recent_bars",
                        lambda s, days=5: pd.DataFrame({"close": [99.0]}))
    out = research._h_options("AAPL.US", "US")
    assert out["expiry"] == "2026-08-21" and out["expiries_total"] == 2
    # put 180 / call 240 = 0.75
    assert out["put_call_oi"] == 0.75
    assert out["atm_strike"] == 100.0, "现价 99 最接近行权价 100"
    assert out["atm_call_iv"] == 0.42


# ---------- 4. 边界：没有的数据要说清楚，不推给别的工具 ----------

def test_fund_flow_hk_is_honest(fake_yf):
    out = research._h_fund_flow("00700.HK", "HK")
    assert "note" in out and "holders" in out["note"]


def test_options_non_us_rejected(fake_yf):
    assert "仅美股" in research._h_options("00700.HK", "HK")["note"]


def test_handlers_never_raise_when_yfinance_dies(monkeypatch):
    """yfinance 抛异常时返回 note，不能把整个报告炸掉。"""
    class _Boom:
        def __init__(self, sym):
            raise RuntimeError("网络炸了")

    mod = types.ModuleType("yfinance")
    mod.Ticker = _Boom
    monkeypatch.setitem(sys.modules, "yfinance", mod)
    rep = research.research("AAPL.US", sections=["profile", "news", "options"])
    for sec in ("profile", "news", "options"):
        assert isinstance(rep[sec], dict)
        assert "note" in rep[sec] or "error" in rep[sec]


# ---------- 5. 默认板块表 ----------

def test_default_sections_have_no_placeholder_only_markets():
    """港美的默认板块必须都是已实现的，不能再有纯占位。"""
    for market in ("HK", "US"):
        for sec in research._DEFAULT_SECTIONS[market]:
            assert sec in research._SECTION_HANDLERS, f"{market} 的 {sec} 没有处理器"
    assert "holders" in research._DEFAULT_SECTIONS["HK"]
    assert "fund_flow" not in research._DEFAULT_SECTIONS["HK"], \
        "港股没有免费资金流接口，不该放进默认板块白跑一次"


def test_format_research_renders_new_sections(fake_yf, monkeypatch):
    monkeypatch.setattr(research, "get_recent_bars",
                        lambda s, days=5: pd.DataFrame({"close": [99.0]}))
    rep = research.research("AAPL.US", sections=["profile", "financial", "research",
                                                 "holders", "sec_filings", "options"])
    text = research.format_research(rep)
    assert "Fake Corp" in text
    assert "市盈率TTM" in text
    assert "目标价" in text
    assert "Big Fund" in text
    assert "10-Q" in text
    assert "put/call" in text
    assert "💡" not in text, "MCP 补全提示的图标不该再出现"
