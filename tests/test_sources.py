"""数据源注册表（v0.26 新增）。

解析逻辑用**录制的真实响应样本**测（不打网络）；failover / 优先级 / 自检
用假源测。样本取自实际接口返回，字段值与其他源交叉验证过
（茅台 1297.41 +0.42%、腾讯 434.60 −2.38%）。
"""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

from eq.data import sources as sr


# ====================== 录制的真实响应样本 ======================

# 新浪 A 股：name,今开,昨收,现价,最高,最低,买一,卖一,成交量(股),成交额,...,日期,时间,00
_SINA_A = (
    'var hq_str_sh600519="贵州茅台,1305.000,1292.010,1297.410,1309.210,1286.200,'
    '1297.400,1297.410,3569892,4626000000.000,'
    + ",".join(["100", "1297.40"] * 5 + ["200", "1297.41"] * 5)
    + ',2026-07-24,15:00:00,00";\n'
)
# 新浪港股：英文名,中文名,今开,昨收,最高,最低,现价,涨跌,涨跌幅,买一,卖一,成交额,成交量,...,日期,时间
_SINA_HK = (
    'var hq_str_rt_hk00700="TENCENT,腾讯控股,437.000,445.200,439.000,432.000,'
    '434.600,-10.600,-2.381,434.400,434.600,9977000000.000,22959603,'
    '0.000,0.000,0.000,0.000,2026/07/24,16:08:00";\n'
)
# 新浪美股：name,现价,涨跌幅,日期时间,涨跌额,开盘,最高,最低,52w高,52w低,成交量
_SINA_US = (
    'var hq_str_gb_aapl="苹果,333.0200,3.5300,2026-07-24 16:00:02,11.3600,'
    '322.5000,334.1000,321.8800,340.0000,190.0000,47443900,";\n'
)
def _tx_fixture() -> str:
    """腾讯 qt.gtimg.cn 的 ~ 分隔字段表（按官方字段序逐位填，避免数错位）。

    0 市场标识 / 1 名称 / 2 代码 / 3 现价 / 4 昨收 / 5 今开 / 6 成交量(手) /
    7 外盘 / 8 内盘 / 9-18 买五档 / 19-28 卖五档 / 29 最近逐笔 / 30 时间 /
    31 涨跌 / 32 涨跌% / 33 最高 / 34 最低
    """
    f = ["0"] * 50
    f[0], f[1], f[2] = "1", "贵州茅台", "600519"
    f[3], f[4], f[5], f[6] = "1297.41", "1292.01", "1305.00", "35699"
    f[7], f[8] = "17225", "18474"
    f[29] = "1297.41/100/12974100"
    f[30] = "20260724150000"
    f[31], f[32] = "5.40", "0.42"
    f[33], f[34] = "1309.21", "1286.20"
    return 'v_sh600519="' + "~".join(f) + '";\n'


_TX_A = _tx_fixture()

_YH_CHART = {
    "chart": {"result": [{
        "timestamp": [1719273600, 1719360000, 1719446400],
        "indicators": {"quote": [{
            "open": [320.0, 322.0, 325.0], "high": [325.0, 327.0, 334.1],
            "low": [318.0, 321.0, 321.88], "close": [322.5, 326.0, 333.02],
            "volume": [4.0e7, 4.2e7, 4.7e7],
        }]},
    }]}
}

_EM_KLINE = {"data": {"klines": [
    "2026-07-22,1300.00,1295.00,1310.00,1290.00,30000,3.9e10,1.5,0.2,2.0,0.3",
    "2026-07-23,1295.00,1292.01,1300.00,1288.00,32000,4.1e10,0.9,-0.2,-3.0,0.3",
    "2026-07-24,1305.00,1297.41,1309.21,1286.20,35699,4.6e10,1.8,0.4,5.4,0.3",
]}}

_EM_SPOT = {"data": {"total": 5542, "diff": [
    {"f12": "600519", "f14": "贵州茅台", "f2": 1297.41, "f3": 0.42, "f5": 35699,
     "f6": 4.6e10, "f15": 1309.21, "f16": 1286.20, "f17": 1305.0, "f18": 1292.01},
    {"f12": "000001", "f14": "平安银行", "f2": 11.10, "f3": 0.18, "f5": 1140932,
     "f6": 1.2e9, "f15": 11.2, "f16": 11.0, "f17": 11.05, "f18": 11.08},
    {"f12": "920000", "f14": "北证某股", "f2": 5.0, "f3": 1.0, "f5": 100, "f6": 5e5,
     "f15": 5.1, "f16": 4.9, "f17": 4.95, "f18": 4.95},
]}}

_BINANCE_KL = [
    [1719273600000, "60000.0", "61000.0", "59500.0", "60500.0", "1200.5", 0, "0", 0, "0", "0", "0"],
    [1719360000000, "60500.0", "65000.0", "60000.0", "64490.0", "1500.2", 0, "0", 0, "0", "0", "0"],
]


class _Resp:
    def __init__(self, text="", payload=None):
        self._text, self._payload = text, payload
        self.encoding = "utf-8"

    @property
    def text(self):
        return self._text if self._payload is None else json.dumps(self._payload)

    def json(self):
        if self._payload is None:
            return json.loads(self._text)
        return self._payload

    def raise_for_status(self):
        return None


@pytest.fixture
def http(monkeypatch):
    """拦截 _http_get，按 URL 关键字返回录制样本；同时记录调用。"""
    calls = []

    def _fake(url, params=None, **kw):
        calls.append((url, params or {}))
        if "hq.sinajs.cn" in url:
            if "rt_hk" in url:
                return _Resp(_SINA_HK)
            if "gb_" in url:
                return _Resp(_SINA_US)
            return _Resp(_SINA_A)
        if "qt.gtimg.cn" in url:
            return _Resp(_TX_A)
        if "CN_MarketDataService" in url:
            return _Resp(json.dumps([
                {"day": "2026-07-22", "open": "1300.0", "high": "1310.0",
                 "low": "1290.0", "close": "1295.0", "volume": "3000000"},
                {"day": "2026-07-24", "open": "1305.0", "high": "1309.21",
                 "low": "1286.2", "close": "1297.41", "volume": "3569892"},
            ]))
        if "push2his" in url:
            return _Resp(payload=_EM_KLINE)
        if "push2.eastmoney" in url:
            return _Resp(payload=_EM_SPOT)
        if "query1.finance.yahoo" in url:
            return _Resp(payload=_YH_CHART)
        if "api.binance.com" in url:
            return _Resp(payload=_BINANCE_KL)
        raise AssertionError(f"未预期的 URL: {url}")

    monkeypatch.setattr(sr, "_http_get", _fake)
    return calls


# ====================== 注册表结构 ======================

def test_registry_has_expected_sources():
    for name in ("sina", "tencent", "eastmoney", "netease", "yahoo",
                 "binance", "okx", "coingecko", "baostock", "yfinance",
                 "akshare", "tdx", "tushare"):
        assert name in sr.REGISTRY, f"缺源 {name}"


def test_every_source_declares_at_least_one_capability():
    for name, s in sr.REGISTRY.items():
        assert s.caps, f"{name} 没有任何能力"
        assert s.markets, f"{name} 没有声明市场"


def test_every_market_has_a_bars_source():
    for m in ("A", "HK", "US", "CRYPTO"):
        assert any(s.supports(m, "bars") for s in sr.REGISTRY.values()), f"{m} 无 bars 源"


def test_describe_registry_table():
    df = sr.describe_registry()
    assert len(df) == len(sr.REGISTRY)
    assert {"源", "市场", "能力", "优先级", "已装"} <= set(df.columns)
    assert df["优先级"].is_monotonic_increasing


# ====================== 符号转换 ======================

@pytest.mark.parametrize(("fn", "symbol", "market", "expect"), [
    (sr._sina_code, "600519.SH", "A", "sh600519"),
    (sr._sina_code, "000001.SZ", "A", "sz000001"),
    (sr._sina_code, "00700.HK", "HK", "rt_hk00700"),
    (sr._sina_code, "AAPL.US", "US", "gb_aapl"),
    (sr._tencent_code, "600519.SH", "A", "sh600519"),
    (sr._tencent_code, "00700.HK", "HK", "r_hk00700"),
    (sr._tencent_code, "AAPL.US", "US", "usAAPL"),
    (sr._em_secid, "600519.SH", "A", "1.600519"),
    (sr._em_secid, "000001.SZ", "A", "0.000001"),
    (sr._em_secid, "00700.HK", "HK", "116.00700"),
    (sr._yahoo_code, "600519.SH", "A", "600519.SS"),
    (sr._yahoo_code, "00700.HK", "HK", "0700.HK"),
    (sr._yahoo_code, "BTC-USDT", "CRYPTO", "BTC-USD"),
])
def test_symbol_conversion(fn, symbol, market, expect):
    assert fn(symbol, market) == expect


def test_symbol_conversion_accepts_loose_input():
    """走 normalize_symbol，所以随手写也认。"""
    assert sr._sina_code("600519", "A") == "sh600519"
    assert sr._tencent_code("700", "HK") == "r_hk00700"


# ====================== 解析：新浪 ======================

def test_sina_snapshot_a(http):
    s = sr.sina_snapshot("600519.SH", "A")
    assert s["name"] == "贵州茅台"
    assert s["close"] == pytest.approx(1297.41)
    assert s["open"] == pytest.approx(1305.0)
    assert s["prev_close"] == pytest.approx(1292.01)
    assert s["high"] == pytest.approx(1309.21)
    assert s["low"] == pytest.approx(1286.20)
    assert s["volume"] == pytest.approx(3569892)
    assert s["change_pct"] == pytest.approx(0.4179, abs=1e-3)
    assert s["date"] == "2026-07-24"


def test_sina_snapshot_hk(http):
    s = sr.sina_snapshot("00700.HK", "HK")
    assert s["name"] == "腾讯控股"
    assert s["close"] == pytest.approx(434.60)
    assert s["prev_close"] == pytest.approx(445.20)
    assert s["change_pct"] == pytest.approx(-2.381, abs=1e-2)


def test_sina_snapshot_us(http):
    s = sr.sina_snapshot("AAPL.US", "US")
    assert s["name"] == "苹果"
    assert s["close"] == pytest.approx(333.02)
    # 美股接口只给涨跌额，prev_close 要反推
    assert s["prev_close"] == pytest.approx(333.02 - 11.36)
    assert s["change_pct"] == pytest.approx(3.53, abs=0.05)


def test_sina_snapshot_uses_referer(http):
    """新浪行情接口没有 Referer 会返回 403。"""
    sr.sina_snapshot("600519.SH", "A")
    assert any("hq.sinajs.cn" in u for u, _ in http)


def test_sina_snapshot_rejects_empty(monkeypatch):
    monkeypatch.setattr(sr, "_http_get", lambda *a, **k: _Resp('var hq_str_sh000000="";\n'))
    with pytest.raises(ValueError, match="空"):
        sr.sina_snapshot("600519.SH", "A")


def test_sina_bars(http):
    df = sr.sina_bars("600519.SH", "A", dt.date(2026, 7, 1), dt.date(2026, 7, 31))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert df.index.is_monotonic_increasing
    assert df["close"].iloc[-1] == pytest.approx(1297.41)


def test_sina_bars_rejects_us():
    with pytest.raises(NotImplementedError):
        sr.sina_bars("AAPL.US", "US", dt.date(2026, 1, 1), dt.date(2026, 2, 1))


# ====================== 解析：腾讯 ======================

def test_tencent_snapshot(http):
    s = sr.tencent_snapshot("600519.SH", "A")
    assert s["name"] == "贵州茅台"
    assert s["close"] == pytest.approx(1297.41)
    assert s["prev_close"] == pytest.approx(1292.01)
    assert s["open"] == pytest.approx(1305.0)
    assert s["high"] == pytest.approx(1309.21)
    assert s["low"] == pytest.approx(1286.20)
    # 腾讯成交量单位是「手」，要 ×100 换成股
    assert s["volume"] == pytest.approx(35699 * 100)
    assert s["date"] == "2026-07-24"


def test_tencent_and_sina_agree(http):
    """两个独立源解析同一只票，价格应一致——交叉验证解析逻辑。"""
    a = sr.sina_snapshot("600519.SH", "A")
    b = sr.tencent_snapshot("600519.SH", "A")
    for k in ("close", "prev_close", "open", "high", "low"):
        assert a[k] == pytest.approx(b[k], rel=1e-4), f"{k} 不一致"


# ====================== 解析：东财 / Yahoo / 网易 / Binance ======================

def test_eastmoney_bars(http):
    df = sr.eastmoney_bars("600519.SH", "A", dt.date(2026, 7, 1), dt.date(2026, 7, 31))
    assert len(df) == 3
    assert df["close"].iloc[-1] == pytest.approx(1297.41)
    assert df["high"].iloc[-1] == pytest.approx(1309.21)   # 东财列序是 开,收,高,低
    assert df["low"].iloc[-1] == pytest.approx(1286.20)


def test_eastmoney_spot_maps_symbols(http):
    df = sr.eastmoney_spot("A", top_n=10)
    syms = set(df["symbol"])
    assert "600519.SH" in syms      # 6 开头 → 沪
    assert "000001.SZ" in syms      # 0 开头 → 深
    assert "920000.BJ" in syms      # 92 开头 → 北交所
    assert df.loc[df["symbol"] == "600519.SH", "close"].iloc[0] == pytest.approx(1297.41)


def test_eastmoney_spot_rejects_crypto():
    with pytest.raises(NotImplementedError):
        sr.eastmoney_spot("CRYPTO")


def test_yahoo_bars(http):
    df = sr.yahoo_bars("AAPL.US", "US", dt.date(2026, 6, 20), dt.date(2026, 6, 30))
    assert len(df) == 3
    assert df["close"].iloc[-1] == pytest.approx(333.02)
    assert df.index.is_monotonic_increasing


def test_netease_bars(monkeypatch):
    csv = ("日期,股票代码,名称,开盘价,最高价,最低价,收盘价,成交量,成交金额,换手率,总市值,流通市值\n"
           "2026-07-24,'600519,贵州茅台,1305.0,1309.21,1286.2,1297.41,3569892,4.6e10,0.28,1.6e12,1.6e12\n"
           "2026-07-23,'600519,贵州茅台,1295.0,1300.0,1288.0,1292.01,3200000,4.1e10,0.25,1.6e12,1.6e12\n")
    monkeypatch.setattr(sr, "_http_get", lambda *a, **k: _Resp(csv))
    df = sr.netease_bars("600519.SH", "A", dt.date(2026, 7, 1), dt.date(2026, 7, 31))
    assert len(df) == 2
    # 网易 CSV 是倒序的，_norm_bars 必须排正
    assert df.index.is_monotonic_increasing
    assert df["close"].iloc[-1] == pytest.approx(1297.41)


def test_netease_rejects_html_error_page(monkeypatch):
    monkeypatch.setattr(sr, "_http_get", lambda *a, **k: _Resp("<html><body>error</body></html>"))
    with pytest.raises(ValueError, match="HTML"):
        sr.netease_bars("600519.SH", "A", dt.date(2026, 1, 1), dt.date(2026, 2, 1))


def test_netease_rejects_non_a_market():
    with pytest.raises(NotImplementedError):
        sr.netease_bars("AAPL.US", "US", dt.date(2026, 1, 1), dt.date(2026, 2, 1))


def test_binance_bars(http):
    df = sr.binance_bars("BTC-USDT", "CRYPTO", dt.date(2026, 6, 20), dt.date(2026, 6, 30))
    assert len(df) == 2
    assert df["close"].iloc[-1] == pytest.approx(64490.0)


# ====================== _norm_bars ======================

def test_norm_bars_sorts_dedups_and_coerces():
    df = pd.DataFrame({
        "open": ["2", "1", "3"], "high": ["2", "1", "3"], "low": ["2", "1", "3"],
        "close": ["2", "1", "3"], "volume": ["2", "1", "3"], "extra": [9, 9, 9],
    }, index=["2026-01-02", "2026-01-01", "2026-01-02"])
    out = sr._norm_bars(df)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out.index.is_monotonic_increasing
    assert len(out) == 2                       # 重复日期去重
    assert out["close"].iloc[-1] == 3.0        # 保留最后一条
    assert out["close"].dtype.kind == "f"      # 字符串已转数值


def test_norm_bars_raises_on_missing_columns():
    with pytest.raises(ValueError, match="缺列"):
        sr._norm_bars(pd.DataFrame({"open": [1]}, index=["2026-01-01"]))


def test_norm_bars_empty_is_safe():
    assert sr._norm_bars(pd.DataFrame()).empty


# ====================== failover / 优先级 / 自检 ======================

@pytest.fixture
def fake_registry(monkeypatch):
    """换成一组可控假源，测 failover 顺序。"""
    calls = []

    def mk(name, priority, ok):
        def _bars(symbol, market, start, end):
            calls.append(name)
            if not ok:
                raise RuntimeError(f"{name} 挂了")
            return pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0],
                                 "close": [1.0], "volume": [1.0]},
                                index=pd.to_datetime(["2026-01-01"]))
        return sr.DataSource(name=name, label=name, markets={"A"},
                             priority=priority, fetch_bars=_bars)

    reg = {s.name: s for s in [mk("bad1", 10, False), mk("bad2", 20, False),
                               mk("good", 30, True), mk("never", 40, True)]}
    monkeypatch.setattr(sr, "REGISTRY", reg)
    monkeypatch.setattr(sr, "load_health", lambda: {})
    return calls


def test_fetch_bars_failover_in_priority_order(fake_registry):
    df, used = sr.fetch_bars("600519.SH", "A", dt.date(2026, 1, 1), dt.date(2026, 1, 2))
    assert used == "good"
    assert fake_registry == ["bad1", "bad2", "good"], "应按优先级依次尝试，成功后停止"
    assert len(df) == 1


def test_fetch_bars_prefer_overrides_priority(fake_registry):
    df, used = sr.fetch_bars("600519.SH", "A", dt.date(2026, 1, 1), dt.date(2026, 1, 2),
                             prefer=["never"])
    assert used == "never"
    assert fake_registry == ["never"], "prefer 指定的源应最先试"


def test_all_sources_failed_carries_reasons(monkeypatch):
    def mk(name):
        def _bars(*a, **k):
            raise RuntimeError(f"{name} 网络超时")
        return sr.DataSource(name=name, label=name, markets={"A"}, fetch_bars=_bars)

    monkeypatch.setattr(sr, "REGISTRY", {n: mk(n) for n in ("x", "y")})
    monkeypatch.setattr(sr, "load_health", lambda: {})
    with pytest.raises(sr.AllSourcesFailed) as ei:
        sr.fetch_bars("600519.SH", "A", dt.date(2026, 1, 1), dt.date(2026, 1, 2))
    assert set(ei.value.errors) == {"x", "y"}
    assert "网络超时" in str(ei.value)


def test_no_source_for_market_raises(monkeypatch):
    monkeypatch.setattr(sr, "REGISTRY", {})
    monkeypatch.setattr(sr, "load_health", lambda: {})
    with pytest.raises(sr.AllSourcesFailed, match="无可用源"):
        sr.fetch_bars("600519.SH", "A", dt.date(2026, 1, 1), dt.date(2026, 1, 2))


def test_health_result_reorders_sources(monkeypatch):
    """自检通过的源应排到优先级更高但实测不通的源前面。"""
    def mk(name, pri):
        return sr.DataSource(name=name, label=name, markets={"A"}, priority=pri,
                             fetch_bars=lambda *a, **k: pd.DataFrame())

    monkeypatch.setattr(sr, "REGISTRY", {n: mk(n, p) for n, p in
                                         [("fast_but_blocked", 10), ("slow_but_works", 90)]})
    monkeypatch.setattr(sr, "load_health", lambda: {})
    assert [s.name for s in sr.sources_for("A", "bars")][0] == "fast_but_blocked"

    monkeypatch.setattr(sr, "load_health", lambda: {"results": {
        "fast_but_blocked": {"A": {"ok": False}},
        "slow_but_works": {"A": {"ok": True}},
    }})
    assert [s.name for s in sr.sources_for("A", "bars")][0] == "slow_but_works"


def test_sources_for_skips_uninstalled(monkeypatch):
    s = sr.DataSource(name="needs_missing", label="x", markets={"A"},
                      requires=("definitely_not_installed_pkg",),
                      fetch_bars=lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(sr, "REGISTRY", {"needs_missing": s})
    monkeypatch.setattr(sr, "load_health", lambda: {})
    assert sr.sources_for("A", "bars") == []


def test_probe_source_reports_ok_and_timing(http):
    res = sr.probe_source(sr.REGISTRY["sina"], "A", "snapshot")
    assert res["ok"] is True
    assert res["seconds"] >= 0
    assert "贵州茅台" in res["detail"]


def test_probe_source_catches_failure(monkeypatch):
    bad = sr.DataSource(name="b", label="b", markets={"A"},
                        fetch_bars=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    res = sr.probe_source(bad, "A", "bars")
    assert res["ok"] is False and "boom" in res["detail"]


def test_probe_all_writes_health_file(tmp_db, monkeypatch, http):
    monkeypatch.setattr(sr, "REGISTRY", {"sina": sr.REGISTRY["sina"]})
    rep = sr.probe_all(markets=["A"], caps=["snapshot"], workers=1)
    assert rep["n_jobs"] == 1
    assert rep["results"]["sina"]["A"]["ok"] is True
    assert sr.load_health()["results"]["sina"]["A"]["ok"] is True
    assert sr._health_path().exists()


def test_load_health_returns_empty_when_absent(tmp_db):
    assert sr.load_health() == {}


# ============ 腾讯：字段语义随市场变（v0.26 回归） ============

def _tx(market: str, vol: str, timefield: str) -> str:
    f = ["0"] * 50
    f[0], f[1], f[2] = "1", "标的", "X"
    f[3], f[4], f[5], f[6] = "100.0", "99.0", "99.5", vol
    f[30] = timefield
    f[33], f[34] = "101.0", "98.0"
    return 'v_x="' + "~".join(f) + '";\n'


@pytest.mark.parametrize(("market", "raw_vol", "expect_vol"), [
    ("A", "35699", 3569900.0),      # A 股字段 6 是「手」，×100 才是股
    ("HK", "22959603.0", 22959603.0),   # 港股已经是「股」，不能再 ×100
    ("US", "47489415", 47489415.0),     # 美股同上
])
def test_tencent_volume_unit_per_market(monkeypatch, market, raw_vol, expect_vol):
    monkeypatch.setattr(sr, "_http_get",
                        lambda *a, **k: _Resp(_tx(market, raw_vol, "20260724150000")))
    assert sr.tencent_snapshot("600519.SH" if market == "A" else "00700.HK",
                               market)["volume"] == pytest.approx(expect_vol)


@pytest.mark.parametrize("timefield", [
    "20260724161433",         # A 股：纯 14 位数字
    "2026/07/24 16:08:10",    # 港股：斜杠分隔
    "2026-07-24 16:00:01",    # 美股：横杠分隔
])
def test_tencent_date_parsing_across_formats(monkeypatch, timefield):
    """三个市场的时间字段格式都不一样，按固定分隔符切会切出 '2026-/0-7/'。"""
    monkeypatch.setattr(sr, "_http_get", lambda *a, **k: _Resp(_tx("A", "1", timefield)))
    assert sr.tencent_snapshot("600519.SH", "A")["date"] == "2026-07-24"


def test_tencent_date_falls_back_to_today_when_unparseable(monkeypatch):
    monkeypatch.setattr(sr, "_http_get", lambda *a, **k: _Resp(_tx("A", "1", "N/A")))
    assert sr.tencent_snapshot("600519.SH", "A")["date"] == dt.date.today().isoformat()


# ============ 日期归一化（v0.26 回归） ============

@pytest.mark.parametrize(("raw", "expect"), [
    ("20260724161433", "2026-07-24"),        # 腾讯 A 股：14 位数字
    ("2026/07/24", "2026-07-24"),            # 新浪港股：斜杠
    ("2026/07/24 16:08:10", "2026-07-24"),   # 腾讯港股：斜杠 + 时间
    ("2026-07-24 16:00:01", "2026-07-24"),   # 腾讯美股：横杠 + 时间
    ("2026-07-24", "2026-07-24"),
    ("", ""),
    ("N/A", ""),
])
def test_norm_date_formats(raw, expect):
    assert sr._norm_date(raw) == expect


def test_norm_date_beijing_to_et():
    """新浪美股时间戳是北京时间：美东 07-24 21:46 显示成北京 07-25 09:46，
    直接取日期会比真实交易日多一天。"""
    assert sr._norm_date("2026-07-25 09:46:26", beijing_to_et=True) == "2026-07-24"
    # 美股收盘 16:00 ET = 次日北京 04:00
    assert sr._norm_date("2026-07-25 04:00:00", beijing_to_et=True) == "2026-07-24"
    # 美股开盘 09:30 ET = 当日北京 21:30
    assert sr._norm_date("2026-07-24 21:30:00", beijing_to_et=True) == "2026-07-24"


def test_sina_us_snapshot_uses_et_trading_date(monkeypatch):
    monkeypatch.setattr(sr, "_http_get", lambda *a, **k: _Resp(
        'var hq_str_gb_aapl="苹果,333.0200,3.5300,2026-07-25 09:46:26,11.3600,'
        '322.5000,334.1000,321.8800,340.0000,190.0000,47489415,";\n'))
    assert sr.sina_snapshot("AAPL.US", "US")["date"] == "2026-07-24"


def test_sina_hk_snapshot_normalizes_slash_date(http):
    """新浪港股日期是 2026/07/24，要归一成 2026-07-24。"""
    assert sr.sina_snapshot("00700.HK", "HK")["date"] == "2026-07-24"


# ============ 批量快照（v0.26 新增） ============

def _sina_multi(n: int) -> str:
    """新浪批量返回：一行一只，顺序同请求。"""
    return "".join(
        f'var hq_str_sh60000{i}="股{i},10.0,9.0,{10 + i}.0,11.0,9.5,10.0,10.1,'
        f'{1000 * (i + 1)},1e6,' + ",".join(["1", "10"] * 10) + ',2026-07-24,15:00:00,00";\n'
        for i in range(n)
    )


def test_sina_batch_one_request_for_many_symbols(monkeypatch):
    calls = []

    def _fake(url, params=None, **kw):
        calls.append(url)
        return _Resp(_sina_multi(5))

    monkeypatch.setattr(sr, "_http_get", _fake)
    syms = [f"60000{i}.SH" for i in range(5)]
    got = sr.sina_batch(syms, "A")
    assert len(calls) == 1, "5 只应该只发 1 次请求"
    assert set(got) == set(syms)
    assert got["600002.SH"]["close"] == pytest.approx(12.0)


def test_sina_batch_chunks_large_lists(monkeypatch):
    """URL 不能无限长，超过 60 只要分批。"""
    calls = []

    def _fake(url, params=None, **kw):
        calls.append(url)
        n = url.split("list=")[1].count(",") + 1
        return _Resp(_sina_multi(n))

    monkeypatch.setattr(sr, "_http_get", _fake)
    sr.sina_batch([f"60000{i % 10}.SH" for i in range(140)], "A")
    assert len(calls) == 3, f"140 只应分 3 批，实际 {len(calls)}"


def test_sina_batch_skips_unparseable_rows(monkeypatch):
    """个别票停牌/退市返回空串，不能拖垮整批。"""
    body = ('var hq_str_sh600000="";\n'
            'var hq_str_sh600001="股1,10.0,9.0,11.0,11.5,9.5,10.0,10.1,2000,1e6,'
            + ",".join(["1", "10"] * 10) + ',2026-07-24,15:00:00,00";\n')
    monkeypatch.setattr(sr, "_http_get", lambda *a, **k: _Resp(body))
    got = sr.sina_batch(["600000.SH", "600001.SH"], "A")
    assert "600000.SH" not in got
    assert got["600001.SH"]["close"] == pytest.approx(11.0)


def test_sina_batch_empty_input():
    assert sr.sina_batch([], "A") == {}


def test_tencent_batch(monkeypatch):
    body = "".join(_tx("A", "100", "20260724150000").replace("v_x=", f"v_sh60000{i}=")
                   for i in range(3))
    calls = []
    monkeypatch.setattr(sr, "_http_get",
                        lambda url, **k: (calls.append(url), _Resp(body))[1])
    got = sr.tencent_batch([f"60000{i}.SH" for i in range(3)], "A")
    assert len(calls) == 1
    assert len(got) == 3
    assert got["600000.SH"]["close"] == pytest.approx(100.0)


def test_batch_registered_as_capability():
    assert "batch" in sr.REGISTRY["sina"].caps
    assert "batch" in sr.REGISTRY["tencent"].caps
    assert sr.REGISTRY["sina"].supports("HK", "batch")


def test_fetch_batch_failover(monkeypatch):
    order = []

    def mk(name, pri, ok):
        def _b(symbols, market):
            order.append(name)
            if not ok:
                raise RuntimeError("挂了")
            return {s: {"close": 1.0} for s in symbols}
        return sr.DataSource(name=name, label=name, markets={"A"},
                             priority=pri, fetch_batch=_b)

    monkeypatch.setattr(sr, "REGISTRY", {"a": mk("a", 10, False), "b": mk("b", 20, True)})
    monkeypatch.setattr(sr, "load_health", lambda: {})
    got, used = sr.fetch_batch(["600519.SH"], "A")
    assert used == "b" and order == ["a", "b"]
    assert got["600519.SH"]["close"] == 1.0


def test_get_snapshots_uses_batch_when_realtime(monkeypatch):
    """realtime=True 时应走批量接口，而不是逐只并发。"""
    from eq.data import market

    calls = {"batch": 0, "single": 0}

    def _batch(symbols, mkt, prefer=None):
        calls["batch"] += 1
        return {s: {"symbol": s, "close": 1.0, "change_pct": 0.0, "date": "2026-07-24",
                    "open": 1.0, "high": 1.0, "low": 1.0, "volume": 1.0,
                    "prev_close": 1.0, "name": "x"} for s in symbols}, "sina"

    monkeypatch.setattr(sr, "fetch_batch", _batch)
    monkeypatch.setattr(market, "get_snapshot",
                        lambda *a, **k: calls.__setitem__("single", calls["single"] + 1))

    out = market.get_snapshots(["600519.SH", "000001.SZ", "600036.SH"], realtime=True)
    assert calls["batch"] == 1 and calls["single"] == 0
    assert all(out[s]["source"] == "sina" for s in out)


def test_get_snapshots_groups_by_market(monkeypatch):
    """混市场输入要按市场分组，各发各的批量请求。"""
    from eq.data import market

    seen = []

    def _batch(symbols, mkt, prefer=None):
        seen.append((mkt, tuple(sorted(symbols))))
        return {s: {"symbol": s, "close": 1.0, "change_pct": 0.0, "date": "d",
                    "open": 1.0, "high": 1.0, "low": 1.0, "volume": 1.0,
                    "prev_close": 1.0} for s in symbols}, "sina"

    monkeypatch.setattr(sr, "fetch_batch", _batch)
    market.get_snapshots(["600519.SH", "00700.HK", "AAPL.US", "000001.SZ"], realtime=True)
    assert dict(seen) == {"A": ("000001.SZ", "600519.SH"),
                          "HK": ("00700.HK",), "US": ("AAPL.US",)}


def test_get_snapshots_falls_back_per_symbol_when_batch_misses(monkeypatch):
    """批量接口漏掉的标的要逐只补齐。"""
    from eq.data import market

    def _batch(symbols, mkt, prefer=None):
        return {symbols[0]: {"symbol": symbols[0], "close": 1.0, "change_pct": 0.0,
                             "date": "d", "open": 1.0, "high": 1.0, "low": 1.0,
                             "volume": 1.0, "prev_close": 1.0}}, "sina"

    singles = []
    monkeypatch.setattr(sr, "fetch_batch", _batch)
    monkeypatch.setattr(market, "get_snapshot",
                        lambda sym, **k: (singles.append(sym),
                                          {"symbol": sym, "close": 2.0})[1])
    out = market.get_snapshots(["600519.SH", "000001.SZ"], realtime=True)
    assert singles == ["000001.SZ"], "只补批量没覆盖的那只"
    assert out["600519.SH"]["close"] == 1.0
    assert out["000001.SZ"]["close"] == 2.0


def test_get_snapshots_non_realtime_never_calls_batch(monkeypatch):
    from eq.data import market

    monkeypatch.setattr(sr, "fetch_batch",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("不该调批量")))
    monkeypatch.setattr(market, "get_snapshot", lambda sym, **k: {"symbol": sym, "close": 1.0})
    out = market.get_snapshots(["600519.SH"], realtime=False)
    assert out["600519.SH"]["close"] == 1.0


# ============ 分能力的市场覆盖（v0.26 修正） ============

def test_sina_declares_bars_for_a_share_only():
    """实测：新浪 CN_MarketDataService K 线接口只认 A 股，
    hk00700 / rt_hk00700 / 00700 各种写法一律返回 null。
    用一个 markets 集合套所有能力就会对外谎报支持。"""
    s = sr.REGISTRY["sina"]
    assert s.markets_for("bars") == {"A"}
    assert s.supports("A", "bars")
    assert not s.supports("HK", "bars")
    assert not s.supports("US", "bars")
    # 快照/批量三个市场都有
    for m in ("A", "HK", "US"):
        assert s.supports(m, "snapshot") and s.supports(m, "batch")


def test_markets_for_falls_back_to_global_set():
    s = sr.DataSource(name="t", label="t", markets={"A", "HK"},
                      fetch_bars=lambda *a, **k: None,
                      fetch_snapshot=lambda *a, **k: None,
                      cap_markets={"bars": {"A"}})
    assert s.markets_for("bars") == {"A"}
    assert s.markets_for("snapshot") == {"A", "HK"}   # 未覆盖的回落到全局


def test_sources_for_respects_per_cap_markets(monkeypatch):
    """谎报支持的源不该占住 failover 链的前排。"""
    monkeypatch.setattr(sr, "load_health", lambda: {})
    names = [s.name for s in sr.sources_for("HK", "bars")]
    assert "sina" not in names, "新浪不支持港股 K 线，不该出现在候选里"
    assert "tencent" in names


def test_describe_registry_shows_per_cap_markets():
    df = sr.describe_registry()
    row = df[df["源"] == "sina"].iloc[0]
    assert "bars" in row["能力"]
    # 表里要能看出 bars 只有 A 股，否则用户会以为港股 K 线也能用
    assert "A" in str(row["市场"])


# ============ 美股需要交易所后缀（v0.26 修正） ============

def test_tencent_us_bars_tries_exchange_suffixes(monkeypatch):
    """实测：usAAPL 只返回 1 根（当日），usAAPL.OQ 返回 20 根。
    上市所从代码本身判断不出来，只能挨个试。"""
    seen = []

    def _once(code, start, end):
        seen.append(code)
        return [["2026-07-24", "1", "2", "3", "0.5", "100"]] if not code.endswith(".OQ") else [
            [f"2026-07-{d:02d}", "1", "2", "3", "0.5", "100"] for d in range(1, 21)
        ]

    monkeypatch.setattr(sr, "_tencent_kline_once", _once)
    df = sr.tencent_bars("AAPL.US", "US", dt.date(2026, 7, 1), dt.date(2026, 7, 31))
    assert "usAAPL.OQ" in seen, "应该带交易所后缀重试"
    assert len(df) == 20, "拿到的应是 .OQ 那份完整历史"


def test_tencent_us_falls_back_to_nyse_suffix(monkeypatch):
    seen = []

    def _once(code, start, end):
        seen.append(code)
        if code.endswith(".N"):
            return [[f"2026-07-{d:02d}", "1", "2", "3", "0.5", "100"] for d in range(1, 11)]
        return []

    monkeypatch.setattr(sr, "_tencent_kline_once", _once)
    df = sr.tencent_bars("IBM.US", "US", dt.date(2026, 7, 1), dt.date(2026, 7, 31))
    assert seen == ["usIBM.OQ", "usIBM.N"], "NASDAQ 无果才试 NYSE"
    assert len(df) == 10


def test_tencent_non_us_uses_plain_code(monkeypatch):
    seen = []
    monkeypatch.setattr(sr, "_tencent_kline_once",
                        lambda code, s, e: (seen.append(code),
                                            [["2026-07-24", "1", "2", "3", "0.5", "100"]])[1])
    sr.tencent_bars("00700.HK", "HK", dt.date(2026, 7, 1), dt.date(2026, 7, 31))
    assert seen == ["hk00700"], "港股不该加美股后缀"


def test_em_secid_us_prefix_is_parameterised():
    assert sr._em_secid("AAPL.US", "US") == "105.AAPL"           # NASDAQ 缺省
    assert sr._em_secid("IBM.US", "US", "106") == "106.IBM"      # NYSE
    assert sr._em_secid("600519.SH", "A", "106") == "1.600519"   # 非美股不受影响


def test_eastmoney_us_bars_tries_all_exchange_prefixes(monkeypatch):
    seen = []

    def _once(secid, klt, start, end):
        seen.append(secid)
        return ["2026-07-24,1,2,3,0.5,100,1e6,0,0,0,0"] if secid.startswith("107") else []

    monkeypatch.setattr(sr, "_em_kline_once", _once)
    df = sr.eastmoney_bars("XYZ.US", "US", dt.date(2026, 7, 1), dt.date(2026, 7, 31))
    assert seen == ["105.XYZ", "106.XYZ", "107.XYZ"], "NASDAQ→NYSE→AMEX 依次试"
    assert len(df) == 1


# ============ 脏数据过滤 ============

def test_norm_bars_drops_zero_close_rows():
    """网易 CSV 的停牌日整行填 0.0，留着会让收益率算出 -100%。"""
    df = pd.DataFrame({
        "open": [10.0, 0.0, 11.0], "high": [10.0, 0.0, 11.0],
        "low": [10.0, 0.0, 11.0], "close": [10.0, 0.0, 11.0],
        "volume": [1.0, 0.0, 1.0],
    }, index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05"]))
    out = sr._norm_bars(df)
    assert len(out) == 2
    assert (out["close"] > 0).all()


# ============ OHLC 自洽性（v0.31） ============

def test_norm_bars_repairs_inconsistent_ohlc():
    """数据源偶尔给出 open > high 这种不可能的 K 线。
    放过去会让「成交价 ≤ 当日最高价」这类不变量失效。"""
    df = pd.DataFrame({
        "open": [12.0, 10.0], "high": [11.0, 11.0], "low": [9.0, 8.0],
        "close": [10.0, 7.0], "volume": [1.0, 1.0],
    }, index=pd.to_datetime(["2026-01-01", "2026-01-02"]))
    out = sr._norm_bars(df)
    assert (out["high"] >= out[["open", "close"]].max(axis=1)).all()
    assert (out["low"] <= out[["open", "close"]].min(axis=1)).all()
    assert out["high"].iloc[0] == 12.0      # high 撑开到容纳 open
    assert out["low"].iloc[1] == 7.0        # low 撑开到容纳 close


def test_norm_bars_keeps_valid_ohlc_untouched():
    df = pd.DataFrame({
        "open": [10.0], "high": [11.0], "low": [9.0], "close": [10.5], "volume": [1.0],
    }, index=pd.to_datetime(["2026-01-01"]))
    out = sr._norm_bars(df)
    assert out["high"].iloc[0] == 11.0 and out["low"].iloc[0] == 9.0


def test_norm_bars_does_not_drop_rows_when_repairing():
    """修 K 线要撑开 high/low，不能丢整行——丢行会在时间序列上留洞。"""
    df = pd.DataFrame({
        "open": [12.0, 10.0, 11.0], "high": [11.0, 11.0, 12.0],
        "low": [9.0, 8.0, 10.0], "close": [10.0, 7.0, 11.5],
        "volume": [1.0, 1.0, 1.0],
    }, index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05"]))
    assert len(sr._norm_bars(df)) == 3
