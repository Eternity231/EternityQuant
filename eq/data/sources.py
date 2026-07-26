"""数据源注册表（v0.26 新增）。

**为什么要有这个模块**

此前数据源是写死在 ``market.py`` 里的两级链：A 股 baostock→akshare，
港/美 yfinance→akshare。问题有三个：

1. 加一个源要改 ``get_recent_bars`` 的 if/else，越加越乱；
2. 某个源在你的网络环境下不通时，只能整段 try/except 往下掉，
   不知道是哪个源挂了、挂多久了；
3. **不同人的网络环境差别极大**——东财/腾讯/新浪在国内直连很快，
   在海外或受限网络可能完全不通；yfinance 在国内常被限流。
   写死优先级注定有人不合适。

所以改成注册表 + 自检：所有源都注册进来，按 ``priority`` 排序自动 failover，
再用 ``eq data sources --test`` 在**你自己的机器上**实测一遍，
把真正通的源排到前面（结果写进 ``.eternityquant/source_health.json``）。

**能力（capability）**

- ``bars``     历史 K 线 → ``DataFrame[open/high/low/close/volume]``，date 索引
- ``snapshot`` 单只实时快照 → dict
- ``spot``     全市场快照列表 → ``DataFrame[symbol/name/close/change_pct/...]``

新增一个源只要写好 fetch 函数并 :func:`register` 一下，不用碰调用方。
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import pandas as pd

logger = logging.getLogger(__name__)

Market = Literal["A", "HK", "US", "CRYPTO"]
Capability = Literal["bars", "snapshot", "spot", "batch"]

_UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
}
_OHLCV = ["open", "high", "low", "close", "volume"]
DEFAULT_TIMEOUT = 10


def _http_get(url: str, params: dict | None = None, *, timeout: int = DEFAULT_TIMEOUT,
              encoding: str | None = None, referer: str | None = None):
    import requests

    headers = dict(_UA)
    if referer:
        headers["Referer"] = referer
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    if encoding:
        r.encoding = encoding
    return r


def _f(v, default: float | None = None) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return default if (x != x) else x  # NaN → default


def _norm_date(raw: str, *, beijing_to_et: bool = False) -> str:
    """把各源五花八门的日期字段统一成 ``YYYY-MM-DD``。

    实测到的格式：``20260724161433``（腾讯 A 股）、``2026/07/24``（新浪港股）、
    ``2026-07-25 09:46:26``（新浪美股）。抽数字取前 8 位即可通吃。

    Args:
        beijing_to_et: 新浪美股的时间戳是**北京时间**——美东 07-24 21:46 会显示成
            北京 07-25 09:46，直接取日期会比真实交易日多一天。减 12 小时换算回
            美东（夏令时 -12、冬令时 -13，但美股 09:30~16:00 的交易时段内
            两者落在同一天，减 12 足够）。
    """
    if not raw:
        return ""
    if beijing_to_et:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return (dt.datetime.strptime(raw.strip(), fmt)
                        - dt.timedelta(hours=12)).date().isoformat()
            except ValueError:
                continue
    digits = "".join(ch for ch in raw if ch.isdigit())[:8]
    if len(digits) == 8:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    return ""


def _norm_bars(df: pd.DataFrame) -> pd.DataFrame:
    """统一成 date 索引 + open/high/low/close/volume 五列、按日期升序。"""
    if df is None or df.empty:
        return pd.DataFrame(columns=_OHLCV)
    missing = [c for c in _OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"数据源返回缺列 {missing}，实际列 {list(df.columns)[:10]}")
    out = df[_OHLCV].copy()
    for c in _OHLCV:
        # 统一成 float64：各源给回来的成交量有的是 int、有的是 str、
        # 有的是 Int64（可空整型），下游做算术会因 dtype 不一致出岔子
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    out.index = pd.to_datetime(out.index)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["close"])
    # 收盘价 ≤ 0 一定是脏数据：网易 CSV 的停牌日整行填 0.0，
    # 留着会让收益率算出 -100% 或 inf
    out = out[out["close"] > 0]

    # OHLC 自洽性：必须满足 low ≤ open,close ≤ high。
    # 13 个数据源质量参差，偶尔会给出 open > high 这种不可能的 K 线；
    # 放过去会让「次日最高价」之类的计算得出比理论上限还好的结果
    # （限价回测里表现为成交价高于当日最高价）。
    # 修法是把 high/low 撑开到能容纳 open/close，而不是丢掉整根——
    # 丢掉会在时间序列上留洞，对滚动指标伤害更大。
    hi = out[["high", "open", "close"]].max(axis=1)
    lo = out[["low", "open", "close"]].min(axis=1)
    bad = (hi > out["high"]) | (lo < out["low"])
    if bad.any():
        logger.debug("修正 %d 根 OHLC 不自洽的 K 线", int(bad.sum()))
        out.loc[:, "high"] = hi
        out.loc[:, "low"] = lo
    return out


# ======================================================================
# 数据源定义
# ======================================================================

@dataclass
class DataSource:
    """一个数据源。fetch 函数缺省为 None 表示不支持该能力。"""

    name: str
    label: str
    markets: set[str]
    priority: int = 50                      # 越小越优先
    requires: tuple[str, ...] = ()          # 需要的第三方包
    needs_key: bool = False                 # 是否需要 token/key
    note: str = ""
    fetch_bars: Callable[..., pd.DataFrame] | None = None
    fetch_snapshot: Callable[..., dict] | None = None
    fetch_spot: Callable[..., pd.DataFrame] | None = None
    fetch_batch: Callable[..., dict] | None = None      # 一次请求拿一批快照
    # 分能力的市场覆盖。真实的源**不是**「支持 X 市场」这么整齐——
    # 新浪的实时快照覆盖 A/HK/US，但它的 K 线接口只有 A 股（港股写法全返回 null）；
    # 腾讯 K 线有 A/HK/US，但快照和 K 线的代码写法还不一样。
    # 用一个 markets 集合套所有能力，就会对外谎报支持、真调用时才失败，
    # 还白白占掉 failover 链的前排位置。
    cap_markets: dict[str, set[str]] = field(default_factory=dict)

    @property
    def caps(self) -> list[str]:
        out = []
        if self.fetch_bars:
            out.append("bars")
        if self.fetch_snapshot:
            out.append("snapshot")
        if self.fetch_spot:
            out.append("spot")
        if self.fetch_batch:
            out.append("batch")
        return out

    def installed(self) -> bool:
        """依赖包是否都装了。"""
        import importlib.util

        for pkg in self.requires:
            try:
                if importlib.util.find_spec(pkg) is None:
                    return False
            except (ImportError, ValueError):
                return False
        return True

    def markets_for(self, cap: str) -> set[str]:
        """某个能力实际覆盖的市场（缺省回落到 :attr:`markets`）。"""
        return self.cap_markets.get(cap, self.markets)

    def supports(self, market: str, cap: str) -> bool:
        return cap in self.caps and market in self.markets_for(cap)


REGISTRY: dict[str, DataSource] = {}


def register(src: DataSource) -> DataSource:
    REGISTRY[src.name] = src
    return src


# ---------------------------------------------------------------- 符号转换

def _split(symbol: str) -> tuple[str, str]:
    """``600519.SH`` → ``("600519", "SH")``。"""
    from eq.data.market import normalize_symbol

    code, _, suffix = normalize_symbol(symbol).partition(".")
    return code, suffix


def _sina_code(symbol: str, market: str) -> str:
    code, suffix = _split(symbol)
    if market == "A":
        return f"{suffix.lower()}{code}"          # sh600519
    if market == "HK":
        return f"rt_hk{code.zfill(5)}"            # rt_hk00700
    return f"gb_{code.lower()}"                   # gb_aapl


def _tencent_code(symbol: str, market: str) -> str:
    code, suffix = _split(symbol)
    if market == "A":
        return f"{suffix.lower()}{code}"          # sh600519
    if market == "HK":
        return f"r_hk{code.zfill(5)}"             # r_hk00700
    return f"us{code.upper()}"                    # usAAPL


def _em_secid(symbol: str, market: str, us_prefix: str = "105") -> str:
    """东财 secid。美股要区分交易所：NASDAQ=105 / NYSE=106 / AMEX=107，
    和腾讯要 ``.OQ``/``.N`` 是同一类问题——从代码本身判断不出上市所，
    只能挨个试（见 :func:`eastmoney_bars`）。"""
    code, suffix = _split(symbol)
    if market == "A":
        return f"{'1' if suffix == 'SH' else '0'}.{code}"
    if market == "HK":
        return f"116.{code.zfill(5)}"
    return f"{us_prefix}.{code.upper()}"


def _yahoo_code(symbol: str, market: str) -> str:
    code, suffix = _split(symbol)
    if market == "A":
        return f"{code}.{'SS' if suffix == 'SH' else 'SZ'}"
    if market == "HK":
        return f"{code.lstrip('0').zfill(4)}.HK"
    if market == "CRYPTO":
        base, _, quote = symbol.partition("-")
        return f"{base}-{'USD' if quote in ('USDT', 'USDC', 'BUSD') else quote}"
    return code.upper()


# ======================================================================
# 1) 新浪财经  —— 实时快照（A/HK/US 同一接口）+ K 线（A/HK）
# ======================================================================

_SINA_HQ = "https://hq.sinajs.cn/list="
_SINA_KLINE = "https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData"
_SINA_REFERER = "https://finance.sina.com.cn"


def _sina_parse_line(line: str, symbol: str, market: str) -> dict[str, Any]:
    """解析新浪返回的一行 ``var hq_str_xxx="...";``。"""
    if '="' not in line:
        raise ValueError("新浪返回格式异常")
    payload = line.split('="', 1)[1].rstrip('";').split(",")
    if len(payload) < 6 or not payload[0]:
        raise ValueError(f"新浪返回空（{symbol} 可能停牌或代码不存在）")

    if market == "A":
        # name, 今开, 昨收, 现价, 最高, 最低, ..., 成交量(股), 成交额, ..., 日期, 时间
        name, o, pc, c, h, low = payload[0], *[_f(x) for x in payload[1:6]]
        vol = _f(payload[8], 0.0)
        date = _norm_date(payload[30] if len(payload) > 30 else "")
    elif market == "HK":
        # 英文名, 中文名, 今开, 昨收, 最高, 最低, 现价, 涨跌, 涨跌幅, ...
        name = payload[1] or payload[0]
        o, pc, h, low, c = (_f(payload[2]), _f(payload[3]), _f(payload[4]),
                            _f(payload[5]), _f(payload[6]))
        vol = _f(payload[12], 0.0) if len(payload) > 12 else 0.0
        date = _norm_date(payload[17] if len(payload) > 17 else "")
    else:  # US
        # name, 现价, 涨跌幅, 日期时间, 涨跌额, 开盘, 最高, 最低, 52w高, 52w低, 成交量
        name, c = payload[0], _f(payload[1])
        chg = _f(payload[4], 0.0) or 0.0
        o, h, low = _f(payload[5]), _f(payload[6]), _f(payload[7])
        vol = _f(payload[10], 0.0) if len(payload) > 10 else 0.0
        pc = (c - chg) if c is not None else None
        # 新浪美股给的是北京时间，要换算回美东才是真实交易日
        date = _norm_date(payload[3] or "", beijing_to_et=True)

    if c is None or c <= 0:
        raise ValueError(f"新浪返回无效价格（{symbol}）")
    pc = pc if pc else c
    return {
        "symbol": symbol, "name": name, "date": date or dt.date.today().isoformat(),
        "open": o or c, "high": h or c, "low": low or c, "close": c,
        "volume": vol or 0.0, "prev_close": pc,
        "change_pct": (c - pc) / pc * 100 if pc else 0.0,
    }


def sina_snapshot(symbol: str, market: str) -> dict[str, Any]:
    """新浪实时快照。A/HK/US 三市场共用一个接口，是最快的免费源之一。"""
    r = _http_get(_SINA_HQ + _sina_code(symbol, market), encoding="gbk",
                  referer=_SINA_REFERER)
    return _sina_parse_line(r.text.strip(), symbol, market)


def sina_batch(symbols: list[str], market: str) -> dict[str, dict[str, Any]]:
    """新浪**批量**快照：一次 HTTP 请求拿几十上百只。

    自选/持仓那种「一屏几十只」的场景，逐只并发要几十次网络往返；
    新浪这个接口支持 ``list=sh600519,sh600036,...`` 一次问完，
    50 只从 50 次请求降到 1 次。
    """
    if not symbols:
        return {}
    out: dict[str, dict[str, Any]] = {}
    # 单次 URL 别太长，分批 60 只
    for i in range(0, len(symbols), 60):
        chunk = symbols[i:i + 60]
        codes = ",".join(_sina_code(s, market) for s in chunk)
        r = _http_get(_SINA_HQ + codes, encoding="gbk", referer=_SINA_REFERER)
        lines = [ln for ln in r.text.split("\n") if '="' in ln]
        # 返回顺序与请求顺序一致，按位置对回去
        for sym, line in zip(chunk, lines, strict=False):
            try:
                out[sym] = _sina_parse_line(line, sym, market)
            except Exception as e:
                logger.debug("新浪批量解析 %s 失败：%s", sym, e)
    return out


def sina_bars(symbol: str, market: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """新浪 K 线。``scale=240`` 是日线，也支持 5/15/30/60 分钟。

    **仅 A 股**：这个 CN_MarketDataService 端点不认港股/美股代码（实测所有写法
    都返回 ``null``），所以注册表里用 ``cap_markets={"bars": {"A"}}`` 收窄了声明。
    """
    if market != "A":
        raise NotImplementedError(
            "新浪 K 线接口只覆盖 A 股——hk00700 / rt_hk00700 / 00700 各种写法实测全返回 null"
        )
    need = max((end - start).days, 30) + 10
    r = _http_get(_SINA_KLINE, params={
        "symbol": _sina_code(symbol, market).replace("rt_", ""),
        "scale": 240, "ma": "no", "datalen": min(need, 1023),
    })
    rows = json.loads(r.text) if r.text.strip().startswith("[") else []
    if not rows:
        raise ValueError(f"新浪 K 线返回空：{symbol}")
    df = pd.DataFrame(rows)
    df["day"] = pd.to_datetime(df["day"])
    df = df.set_index("day")
    df = df.rename(columns={"volume": "volume"})
    df = _norm_bars(df)
    return df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]


register(DataSource(
    name="sina", label="新浪财经", markets={"A", "HK", "US"}, priority=10,
    # 实测：CN_MarketDataService 这个 K 线接口只认 A 股，
    # hk00700 / rt_hk00700 / 00700 各种写法一律返回 null
    cap_markets={"bars": {"A"}},
    note="免费无 key，快照 A/HK/US 同一接口且最快；K 线仅 A 股",
    fetch_snapshot=sina_snapshot, fetch_bars=sina_bars, fetch_batch=sina_batch,
))


# ======================================================================
# 2) 腾讯财经 —— 实时快照（A/HK/US）+ 日 K（A/HK/US）
# ======================================================================

_TX_HQ = "https://qt.gtimg.cn/q="
_TX_KLINE = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"


def _tencent_parse_line(line: str, symbol: str, market: str) -> dict[str, Any]:
    """解析腾讯返回的一行 ``v_xxx="a~b~c...";``。"""
    if '="' not in line:
        raise ValueError("腾讯返回格式异常")
    p = line.split('="', 1)[1].rstrip('";').split("~")
    if len(p) < 40 or not p[1]:
        raise ValueError(f"腾讯返回空（{symbol}）")
    name, c, pc, o = p[1], _f(p[3]), _f(p[4]), _f(p[5])
    h, low = _f(p[33]), _f(p[34])
    # 成交量单位随市场变：A 股字段 6 是「手」（×100 才是股），
    # 港股/美股字段 6 已经是「股」。实测：茅台 [6]=35699 手 = 3569892 股，
    # 腾讯控股 [6]=22959603.0 本身就是股数。
    vol = _f(p[6], 0.0) or 0.0
    if market == "A":
        vol *= 100
    # 时间字段格式随市场变（A 股 14 位数字 / 港股斜杠 / 美股横杠），交给 _norm_date
    date = _norm_date(p[30] if len(p) > 30 else "")
    if c is None or c <= 0:
        raise ValueError(f"腾讯返回无效价格（{symbol}）")
    pc = pc or c
    return {
        "symbol": symbol, "name": name,
        "date": date or dt.date.today().isoformat(),
        "open": o or c, "high": h or c, "low": low or c, "close": c,
        "volume": vol, "prev_close": pc,
        "change_pct": (c - pc) / pc * 100 if pc else 0.0,
    }


def tencent_snapshot(symbol: str, market: str) -> dict[str, Any]:
    r = _http_get(_TX_HQ + _tencent_code(symbol, market), encoding="gbk")
    return _tencent_parse_line(r.text.strip(), symbol, market)


def tencent_batch(symbols: list[str], market: str) -> dict[str, dict[str, Any]]:
    """腾讯**批量**快照：一次请求拿一批（同 :func:`sina_batch` 的动机）。"""
    if not symbols:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for i in range(0, len(symbols), 60):
        chunk = symbols[i:i + 60]
        r = _http_get(_TX_HQ + ",".join(_tencent_code(x, market) for x in chunk),
                      encoding="gbk")
        lines = [ln for ln in r.text.split("\n") if '="' in ln]
        for sym, line in zip(chunk, lines, strict=False):
            try:
                out[sym] = _tencent_parse_line(line, sym, market)
            except Exception as e:
                logger.debug("腾讯批量解析 %s 失败：%s", sym, e)
    return out


def _tencent_kline_once(code: str, start: dt.date, end: dt.date) -> list:
    r = _http_get(_TX_KLINE, params={
        "param": f"{code},day,{start.isoformat()},{end.isoformat()},640,qfq"
    })
    j = r.json()
    if j.get("code") != 0:
        raise ValueError(f"腾讯 K 线错误码 {j.get('code')}")
    node = (j.get("data") or {}).get(code) or {}
    return node.get("qfqday") or node.get("day") or []


def tencent_bars(symbol: str, market: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """腾讯日 K。

    美股必须带**交易所后缀**才拿得到历史：实测 ``usAAPL`` 只返回 1 根（当日），
    ``usAAPL.OQ``（NASDAQ）返回 20 根。``.N`` 是 NYSE。
    上市所无法从代码本身判断，所以两个都试，取拿到 bar 多的那个。
    """
    code = _tencent_code(symbol, market).replace("r_hk", "hk")
    if market == "US":
        best: list = []
        for suffix in (".OQ", ".N"):
            try:
                got = _tencent_kline_once(code + suffix, start, end)
            except Exception:
                continue
            if len(got) > len(best):
                best = got
            if len(best) > 1:      # 拿到真实历史就不用再试另一个所了
                break
        bars = best
    else:
        bars = _tencent_kline_once(code, start, end)
    if not bars:
        raise ValueError(f"腾讯 K 线返回空：{symbol}")
    rows = [{"date": b[0], "open": b[1], "close": b[2], "high": b[3],
             "low": b[4], "volume": b[5]} for b in bars if len(b) >= 6]
    df = pd.DataFrame(rows).set_index("date")
    return _norm_bars(df)


register(DataSource(
    name="tencent", label="腾讯财经", markets={"A", "HK", "US"}, priority=15,
    note="免费无 key，A/HK/US 全覆盖；日 K 前复权，国内直连快",
    fetch_snapshot=tencent_snapshot, fetch_bars=tencent_bars, fetch_batch=tencent_batch,
))


# ======================================================================
# 3) 东方财富 —— push2his K 线 + push2 全市场快照
# ======================================================================

_EM_HIS = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
_EM_LIST = "https://push2.eastmoney.com/api/qt/clist/get"
_EM_UT = "fa5fd1943c7b386f172d6893dbfba10b"
_EM_FS = {
    "A": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048",
    "HK": "m:128+t:3,m:128+t:4,m:128+t:1,m:128+t:2",
    "US": "m:105,m:106,m:107",
}


def _em_kline_once(secid: str, klt: str, start: dt.date, end: dt.date) -> list:
    r = _http_get(_EM_HIS, params={
        "secid": secid, "ut": _EM_UT,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": klt, "fqt": "1", "beg": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"), "lmt": "8000",
    })
    return (r.json().get("data") or {}).get("klines") or []


def eastmoney_bars(symbol: str, market: str, start: dt.date, end: dt.date,
                   period: str = "daily") -> pd.DataFrame:
    klt = {"daily": "101", "weekly": "102", "monthly": "103",
           "1": "1", "5": "5", "15": "15", "30": "30", "60": "60"}.get(period, "101")
    if market == "US":
        # 105=NASDAQ / 106=NYSE / 107=AMEX，代码本身看不出上市所，挨个试
        kl = []
        for pfx in ("105", "106", "107"):
            kl = _em_kline_once(_em_secid(symbol, market, pfx), klt, start, end)
            if kl:
                break
    else:
        kl = _em_kline_once(_em_secid(symbol, market), klt, start, end)
    if not kl:
        raise ValueError(f"东财 K 线返回空：{symbol}")
    rows = []
    for line in kl:
        p = line.split(",")
        if len(p) >= 6:
            rows.append({"date": p[0], "open": p[1], "close": p[2],
                         "high": p[3], "low": p[4], "volume": p[5]})
    return _norm_bars(pd.DataFrame(rows).set_index("date"))


def eastmoney_spot(market: str, top_n: int = 100) -> pd.DataFrame:
    """东财全市场快照。A 股一次能拉 5500+ 只，是扫描的理想源。"""
    fs = _EM_FS.get(market)
    if fs is None:
        raise NotImplementedError(f"东财全市场快照不支持 {market}")
    r = _http_get(_EM_LIST, params={
        "pn": 1, "pz": max(top_n, 1), "po": 1, "np": 1, "fltt": 2, "invt": 2,
        "fid": "f3", "fs": fs, "ut": _EM_UT,
        "fields": "f2,f3,f4,f5,f6,f12,f14,f15,f16,f17,f18",
    })
    d = r.json().get("data") or {}
    diff = d.get("diff") or []
    if not diff:
        raise ValueError(f"东财 {market} 快照返回空")
    from eq.data.market import normalize_symbol

    rows = []
    for it in diff:
        code = str(it.get("f12", ""))
        if market == "A":
            # 直接复用 normalize_symbol，别在这儿重写一遍板块判断——
            # 自己写的那版把 920xxx（北交所）判成了 .SH
            sym = normalize_symbol(code)
        elif market == "HK":
            sym = f"{code.zfill(5)}.HK"
        else:
            sym = f"{code.upper()}.US"
        rows.append({
            "symbol": sym, "name": it.get("f14", ""),
            "close": _f(it.get("f2")), "change_pct": _f(it.get("f3")),
            "volume": _f(it.get("f5")), "amount": _f(it.get("f6")),
            "high": _f(it.get("f15")), "low": _f(it.get("f16")),
            "open": _f(it.get("f17")), "prev_close": _f(it.get("f18")),
        })
    return pd.DataFrame(rows)


def eastmoney_snapshot(symbol: str, market: str) -> dict[str, Any]:
    """东财单只快照：借 K 线接口取最近两根（比 clist 精确到单只更省事）。"""
    end = dt.date.today()
    df = eastmoney_bars(symbol, market, end - dt.timedelta(days=20), end)
    if df.empty:
        raise ValueError(f"东财快照无数据：{symbol}")
    last = df.iloc[-1]
    pc = float(df.iloc[-2]["close"]) if len(df) >= 2 else float(last["close"])
    c = float(last["close"])
    return {"symbol": symbol, "name": "", "date": str(df.index[-1].date()),
            "open": float(last["open"]), "high": float(last["high"]),
            "low": float(last["low"]), "close": c, "volume": float(last["volume"]),
            "prev_close": pc, "change_pct": (c - pc) / pc * 100 if pc else 0.0}


register(DataSource(
    name="eastmoney", label="东方财富", markets={"A", "HK", "US"}, priority=20,
    note="免费无 key，A/HK/US K 线 + 全市场快照（A 股 5500+ 只一次拉完）",
    fetch_bars=eastmoney_bars, fetch_spot=eastmoney_spot, fetch_snapshot=eastmoney_snapshot,
))


# ======================================================================
# 4) 网易财经 —— A 股全历史 CSV（含换手率/市值，1990 年至今）
# ======================================================================

_NETEASE = "https://quotes.money.163.com/service/chddata.html"


def netease_bars(symbol: str, market: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    if market != "A":
        raise NotImplementedError("网易历史 CSV 只覆盖 A 股")
    code, suffix = _split(symbol)
    prefix = "0" if suffix == "SH" else "1"      # 沪 0 / 深 1
    r = _http_get(_NETEASE, params={
        "code": prefix + code,
        "start": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"),
        "fields": "TOPEN;HIGH;LOW;TCLOSE;VOTURNOVER;VATURNOVER;TURNOVER;TCAP;MCAP",
    }, encoding="gbk", timeout=20)
    text = r.text.strip()
    if text[:100].lower().lstrip().startswith("<"):
        raise ValueError("网易返回 HTML 错误页（代码可能不存在）")
    from io import StringIO

    df = pd.read_csv(StringIO(text))
    if df.empty:
        raise ValueError(f"网易返回空：{symbol}")
    df = df.rename(columns={"日期": "date", "开盘价": "open", "最高价": "high",
                            "最低价": "low", "收盘价": "close", "成交量": "volume"})
    df = df.set_index("date")
    # 网易 CSV 是倒序的，_norm_bars 会排序
    return _norm_bars(df)


register(DataSource(
    name="netease", label="网易财经", markets={"A"}, priority=35,
    note="A 股全历史 CSV（1990 至今），额外带换手率/总市值/流通市值",
    fetch_bars=netease_bars,
))


# ======================================================================
# 5) Yahoo Finance —— chart/quote API 直连（不经 yfinance，省依赖省开销）
# ======================================================================

_YH_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/"


def yahoo_bars(symbol: str, market: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    r = _http_get(_YH_CHART + _yahoo_code(symbol, market), params={
        "period1": int(dt.datetime.combine(start, dt.time()).timestamp()),
        "period2": int(dt.datetime.combine(end + dt.timedelta(days=1), dt.time()).timestamp()),
        "interval": "1d",
    }, timeout=15)
    res = (r.json().get("chart") or {}).get("result") or []
    if not res:
        raise ValueError(f"Yahoo 返回空：{symbol}")
    node = res[0]
    q = (node.get("indicators") or {}).get("quote") or [{}]
    quote = q[0]
    df = pd.DataFrame({
        "open": quote.get("open"), "high": quote.get("high"),
        "low": quote.get("low"), "close": quote.get("close"),
        "volume": quote.get("volume"),
    }, index=pd.to_datetime(node.get("timestamp") or [], unit="s"))
    if df.empty:
        raise ValueError(f"Yahoo 返回空：{symbol}")
    df.index = df.index.normalize()
    return _norm_bars(df)


def yahoo_snapshot(symbol: str, market: str) -> dict[str, Any]:
    end = dt.date.today()
    df = yahoo_bars(symbol, market, end - dt.timedelta(days=20), end)
    if df.empty:
        raise ValueError(f"Yahoo 快照无数据：{symbol}")
    last = df.iloc[-1]
    c = float(last["close"])
    pc = float(df.iloc[-2]["close"]) if len(df) >= 2 else c
    return {"symbol": symbol, "name": "", "date": str(df.index[-1].date()),
            "open": float(last["open"]), "high": float(last["high"]),
            "low": float(last["low"]), "close": c, "volume": float(last["volume"]),
            "prev_close": pc, "change_pct": (c - pc) / pc * 100 if pc else 0.0}


register(DataSource(
    name="yahoo", label="Yahoo Finance(直连)", markets={"A", "HK", "US", "CRYPTO"}, priority=30,
    note="全球覆盖最广；国内直连常被限流，海外网络下是首选",
    fetch_bars=yahoo_bars, fetch_snapshot=yahoo_snapshot,
))


# ======================================================================
# 6) Binance / CoinGecko —— 加密
# ======================================================================

def binance_bars(symbol: str, market: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    base, _, quote = symbol.partition("-")
    pair = f"{base}{quote or 'USDT'}".upper()
    r = _http_get("https://api.binance.com/api/v3/klines", params={
        "symbol": pair, "interval": "1d",
        "startTime": int(dt.datetime.combine(start, dt.time()).timestamp() * 1000),
        "endTime": int(dt.datetime.combine(end, dt.time()).timestamp() * 1000),
        "limit": 1000,
    }, timeout=15)
    j = r.json()
    if not isinstance(j, list) or not j:
        raise ValueError(f"Binance 返回空：{pair}")
    df = pd.DataFrame([{"date": pd.to_datetime(b[0], unit="ms"), "open": b[1], "high": b[2],
                        "low": b[3], "close": b[4], "volume": b[5]} for b in j]).set_index("date")
    df.index = df.index.normalize()
    return _norm_bars(df)


def binance_spot(market: str, top_n: int = 100) -> pd.DataFrame:
    r = _http_get("https://api.binance.com/api/v3/ticker/24hr", timeout=20)
    rows = []
    for t in r.json():
        sym = t.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        rows.append({
            "symbol": f"{sym[:-4]}-USDT", "name": sym[:-4],
            "close": _f(t.get("lastPrice")), "change_pct": _f(t.get("priceChangePercent")),
            "volume": _f(t.get("volume")), "amount": _f(t.get("quoteVolume")),
            "high": _f(t.get("highPrice")), "low": _f(t.get("lowPrice")),
            "open": _f(t.get("openPrice")), "prev_close": _f(t.get("prevClosePrice")),
        })
    if not rows:
        raise ValueError("Binance 全市场返回空")
    return pd.DataFrame(rows).sort_values("amount", ascending=False).head(top_n)


register(DataSource(
    name="binance", label="Binance", markets={"CRYPTO"}, priority=10,
    note="加密现货，无 key，K 线 + 全市场 24h 快照",
    fetch_bars=binance_bars, fetch_spot=binance_spot,
))


def okx_spot(market: str, top_n: int = 100) -> pd.DataFrame:
    import ccxt

    tickers = ccxt.okx().fetch_tickers()
    rows = []
    for sym, t in tickers.items():
        if not sym.endswith("/USDT") or t.get("percentage") is None:
            continue
        last = _f(t.get("last"), 0.0) or 0.0
        vol = _f(t.get("baseVolume"), 0.0) or 0.0
        rows.append({"symbol": sym.replace("/", "-"), "name": sym.split("/")[0],
                     "close": last, "change_pct": _f(t.get("percentage")),
                     "volume": vol, "amount": last * vol,
                     "high": _f(t.get("high")), "low": _f(t.get("low")),
                     "open": _f(t.get("open")), "prev_close": _f(t.get("previousClose"))})
    if not rows:
        raise ValueError("OKX 返回空")
    return pd.DataFrame(rows).sort_values("amount", ascending=False).head(top_n)


register(DataSource(
    name="okx", label="OKX(ccxt)", markets={"CRYPTO"}, priority=20,
    requires=("ccxt",), note="加密全市场，1200+ 交易对",
    fetch_spot=okx_spot,
))


def coingecko_snapshot(symbol: str, market: str) -> dict[str, Any]:
    base = symbol.partition("-")[0].lower()
    alias = {"btc": "bitcoin", "eth": "ethereum", "bnb": "binancecoin",
             "sol": "solana", "xrp": "ripple", "doge": "dogecoin", "ada": "cardano"}
    cid = alias.get(base, base)
    r = _http_get("https://api.coingecko.com/api/v3/simple/price", params={
        "ids": cid, "vs_currencies": "usd", "include_24hr_change": "true",
        "include_24hr_vol": "true",
    }, timeout=15)
    d = (r.json() or {}).get(cid)
    if not d:
        raise ValueError(f"CoinGecko 无此币种：{symbol}（试试全称如 bitcoin-USDT）")
    c = _f(d.get("usd"))
    chg = _f(d.get("usd_24h_change"), 0.0) or 0.0
    pc = c / (1 + chg / 100) if c and chg != -100 else c
    return {"symbol": symbol, "name": cid, "date": dt.date.today().isoformat(),
            "open": pc, "high": c, "low": c, "close": c,
            "volume": _f(d.get("usd_24h_vol"), 0.0) or 0.0,
            "prev_close": pc, "change_pct": chg}


register(DataSource(
    name="coingecko", label="CoinGecko", markets={"CRYPTO"}, priority=40,
    note="加密现价，免费无 key（有频率限制）",
    fetch_snapshot=coingecko_snapshot,
))


# ======================================================================
# 7) 既有的库类数据源（包一层进注册表，统一 failover）
# ======================================================================

def baostock_bars(symbol: str, market: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    if market != "A":
        raise NotImplementedError("baostock 只覆盖 A 股")
    from eq.data.market import _fetch_baostock_a

    return _norm_bars(_fetch_baostock_a(symbol, start, end))


register(DataSource(
    name="baostock", label="BaoStock", markets={"A"}, priority=25,
    requires=("baostock",), note="A 股日线，TCP 直连无 IP 限流；进程内已串行化",
    fetch_bars=baostock_bars,
))


def yfinance_bars(symbol: str, market: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    from eq.data.market import _fetch_yfinance

    return _norm_bars(_fetch_yfinance(symbol, market, start, end))


register(DataSource(
    name="yfinance", label="yfinance", markets={"A", "HK", "US", "CRYPTO"}, priority=45,
    requires=("yfinance",), note="全球覆盖；国内网络常被限流（429）",
    fetch_bars=yfinance_bars,
))


def akshare_bars(symbol: str, market: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    from eq.data.market import _fetch_akshare_a, _fetch_akshare_fallback

    if market == "A":
        return _norm_bars(_fetch_akshare_a(symbol, start, end))
    return _norm_bars(_fetch_akshare_fallback(symbol, market, start, end))


def akshare_spot(market: str, top_n: int = 100) -> pd.DataFrame:
    from eq.core import scanner

    fn = {"A": scanner.scan_a_share, "HK": scanner.scan_hk, "US": scanner.scan_us}.get(market)
    if fn is None:
        raise NotImplementedError(f"akshare 全市场快照不支持 {market}")
    return fn(top_n=top_n)


register(DataSource(
    name="akshare", label="AkShare", markets={"A", "HK", "US"}, priority=60,
    requires=("akshare",), note="接口最全的兜底源；上游接口偶尔变动",
    fetch_bars=akshare_bars, fetch_spot=akshare_spot,
))


def tdx_bars(symbol: str, market: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """通达信协议（mootdx）。二进制协议、无 HTTP 限流，国内速度极快。"""
    if market != "A":
        raise NotImplementedError("通达信源只覆盖 A 股")
    from mootdx.quotes import Quotes

    code, _ = _split(symbol)
    need = max((end - start).days, 60)
    df = Quotes.factory(market="std").bars(symbol=code, frequency=9,
                                           offset=min(need, 800))
    if df is None or df.empty:
        raise ValueError(f"通达信返回空：{symbol}")
    df = df.rename(columns={"vol": "volume", "amount": "amount"})
    if "datetime" in df.columns:
        df = df.set_index("datetime")
    df.index = pd.to_datetime(df.index)
    out = _norm_bars(df)
    return out.loc[(out.index >= pd.Timestamp(start)) & (out.index <= pd.Timestamp(end))]


register(DataSource(
    name="tdx", label="通达信(mootdx)", markets={"A"}, priority=40,
    requires=("mootdx",), note="TDX 二进制协议，无 HTTP 限流；需能连通达信行情服务器",
    fetch_bars=tdx_bars,
))


def tushare_bars(symbol: str, market: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Tushare Pro。需在 .eternityquant/.env 配 ``TUSHARE_TOKEN``。"""
    import os

    from eq.core.python_dotenv_loader import load_dotenv_if_present

    load_dotenv_if_present()
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("未配置 TUSHARE_TOKEN（在 .eternityquant/.env 里加一行）")
    if market != "A":
        raise NotImplementedError("此处的 Tushare 封装只做 A 股日线")
    import tushare as ts

    pro = ts.pro_api(token)
    code, suffix = _split(symbol)
    df = pro.daily(ts_code=f"{code}.{suffix}", start_date=start.strftime("%Y%m%d"),
                   end_date=end.strftime("%Y%m%d"))
    if df is None or df.empty:
        raise ValueError(f"Tushare 返回空：{symbol}")
    df = df.rename(columns={"trade_date": "date", "vol": "volume"}).set_index("date")
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    return _norm_bars(df)


register(DataSource(
    name="tushare", label="Tushare Pro", markets={"A"}, priority=70,
    requires=("tushare",), needs_key=True,
    note="需 TUSHARE_TOKEN；积分够的话有财务/龙虎榜等全套数据",
    fetch_bars=tushare_bars,
))


# ======================================================================
# 统一取数入口（按优先级 failover）
# ======================================================================

_HEALTH_FILE = "source_health.json"


def _health_path():
    from eq.db import DEFAULT_HOME

    return DEFAULT_HOME / _HEALTH_FILE


def load_health() -> dict[str, Any]:
    """读 ``eq data sources --test`` 存下来的自检结果。"""
    try:
        p = _health_path()
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug("读取源健康度失败：%s", e)
    return {}


def save_health(report: dict[str, Any]) -> None:
    try:
        p = _health_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.debug("写入源健康度失败：%s", e)


def sources_for(market: str, cap: str, prefer: list[str] | None = None) -> list[DataSource]:
    """列出支持 ``(market, cap)`` 的源，按「自检可用 → priority」排序。

    自检结果（``eq data sources --test``）优先：本机实测通的源排前面、
    实测不通的排后面。没跑过自检就纯按 priority。
    """
    cands = [s for s in REGISTRY.values() if s.supports(market, cap) and s.installed()]
    health = load_health().get("results", {})

    def _key(s: DataSource):
        h = health.get(s.name, {}).get(market, {})
        # 0=实测通 / 1=没测过 / 2=实测不通
        rank = 0 if h.get("ok") else (2 if h.get("ok") is False else 1)
        return (rank, s.priority, s.name)

    cands.sort(key=_key)
    if prefer:
        order = {n: i for i, n in enumerate(prefer)}
        cands.sort(key=lambda s: order.get(s.name, 10_000))
    return cands


class AllSourcesFailed(RuntimeError):
    """所有候选源都失败。``errors`` 里是每个源的失败原因。"""

    def __init__(self, market: str, cap: str, errors: dict[str, str]):
        self.errors = errors
        detail = "；".join(f"{k}: {v}" for k, v in list(errors.items())[:6]) or "无可用源"
        super().__init__(f"{market} 的 {cap} 全部数据源失败（{len(errors)} 个）→ {detail}")


def fetch_bars(symbol: str, market: str, start: dt.date, end: dt.date,
               prefer: list[str] | None = None) -> tuple[pd.DataFrame, str]:
    """按优先级依次尝试各源拿日线。返回 ``(df, 实际用的源名)``。"""
    errors: dict[str, str] = {}
    for src in sources_for(market, "bars", prefer):
        try:
            df = src.fetch_bars(symbol, market, start, end)
            if df is not None and not df.empty:
                logger.debug("bars %s ← %s（%d 行）", symbol, src.name, len(df))
                return df, src.name
            errors[src.name] = "返回空"
        except Exception as e:
            errors[src.name] = f"{type(e).__name__}: {str(e)[:80]}"
    raise AllSourcesFailed(market, "bars", errors)


def fetch_snapshot(symbol: str, market: str,
                   prefer: list[str] | None = None) -> tuple[dict[str, Any], str]:
    """按优先级依次尝试各源拿实时快照。返回 ``(snapshot, 源名)``。"""
    errors: dict[str, str] = {}
    for src in sources_for(market, "snapshot", prefer):
        try:
            snap = src.fetch_snapshot(symbol, market)
            if snap and snap.get("close"):
                return snap, src.name
            errors[src.name] = "返回空"
        except Exception as e:
            errors[src.name] = f"{type(e).__name__}: {str(e)[:80]}"
    raise AllSourcesFailed(market, "snapshot", errors)


def fetch_batch(symbols: list[str], market: str,
                prefer: list[str] | None = None) -> tuple[dict[str, dict[str, Any]], str]:
    """一次请求拿一批快照。新浪/腾讯支持，能把 N 次网络往返压成 1 次。

    只要**有一只**解析成功就算这个源可用（个别停牌/退市票拿不到很正常）。
    """
    errors: dict[str, str] = {}
    for src in sources_for(market, "batch", prefer):
        try:
            got = src.fetch_batch(symbols, market)
            if got:
                return got, src.name
            errors[src.name] = "返回空"
        except Exception as e:
            errors[src.name] = f"{type(e).__name__}: {str(e)[:80]}"
    raise AllSourcesFailed(market, "batch", errors)


def fetch_spot(market: str, top_n: int = 100,
               prefer: list[str] | None = None) -> tuple[pd.DataFrame, str]:
    """按优先级依次尝试各源拿全市场快照。返回 ``(df, 源名)``。"""
    errors: dict[str, str] = {}
    for src in sources_for(market, "spot", prefer):
        try:
            df = src.fetch_spot(market, top_n)
            if df is not None and not df.empty:
                return df, src.name
            errors[src.name] = "返回空"
        except Exception as e:
            errors[src.name] = f"{type(e).__name__}: {str(e)[:80]}"
    raise AllSourcesFailed(market, "spot", errors)


# ======================================================================
# 自检
# ======================================================================

_PROBE_SYMBOL = {"A": "600519.SH", "HK": "00700.HK", "US": "AAPL.US", "CRYPTO": "BTC-USDT"}


def probe_source(src: DataSource, market: str, cap: str,
                 symbol: str | None = None) -> dict[str, Any]:
    """实测单个源的单个能力。返回 ``{ok, seconds, detail}``。"""
    sym = symbol or _PROBE_SYMBOL.get(market, "600519.SH")
    t0 = time.time()
    try:
        if cap == "bars":
            end = dt.date.today()
            df = src.fetch_bars(sym, market, end - dt.timedelta(days=30), end)
            detail = f"{len(df)} 根  {df.index[0].date()}~{df.index[-1].date()}  收 {df['close'].iloc[-1]:.4g}"
            ok = len(df) > 0
        elif cap == "snapshot":
            s = src.fetch_snapshot(sym, market)
            detail = f"{s.get('name') or sym}  收 {s['close']:.4g}  {s['change_pct']:+.2f}%"
            ok = bool(s.get("close"))
        else:
            df = src.fetch_spot(market, 20)
            detail = f"{len(df)} 只  首条 {df.iloc[0]['symbol']}"
            ok = len(df) > 0
        return {"ok": ok, "seconds": round(time.time() - t0, 2), "detail": detail}
    except Exception as e:
        return {"ok": False, "seconds": round(time.time() - t0, 2),
                "detail": f"{type(e).__name__}: {str(e)[:110]}"}


def probe_all(markets: list[str] | None = None, caps: list[str] | None = None,
              workers: int = 6,
              on_result: Callable[[str, str, str, dict], None] | None = None) -> dict[str, Any]:
    """并发实测全部源 × 市场 × 能力。

    这是本模块的核心用法：**在你自己的机器上**跑一遍，看哪些源真的通。
    结果落盘到 ``.eternityquant/source_health.json``，之后 :func:`sources_for`
    会自动把实测通的源排到前面。
    """
    import contextlib
    import io
    import sys
    from concurrent.futures import ThreadPoolExecutor

    markets = markets or ["A", "HK", "US", "CRYPTO"]
    caps = caps or ["bars", "snapshot", "spot"]

    jobs = [
        (src, m, c)
        for src in REGISTRY.values() if src.installed()
        for m in markets if m in src.markets
        for c in caps if c in src.caps
    ]

    def _run(job):
        src, m, c = job
        return src.name, m, c, probe_source(src, m, c)

    results: dict[str, Any] = {}
    # akshare 会打 tqdm 进度条、baostock 会打 "login success!"，
    # 这些库自带的输出会把自检报告搅成一团。整段重定向掉，
    # 只把 on_result 的那一行放回真正的 stdout。
    real_stdout = sys.stdout
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for name, m, c, res in pool.map(_run, jobs):
                results.setdefault(name, {}).setdefault(m, {})
                # 一个 (源, 市场) 下多个能力：只要有一个通就算这个市场可用
                results[name][m][c] = res
                prev = results[name][m].get("ok")
                results[name][m]["ok"] = bool(res["ok"]) or bool(prev)
                if on_result:
                    with contextlib.redirect_stdout(real_stdout):
                        on_result(name, m, c, res)

    report = {
        "tested_at": dt.datetime.now().isoformat(timespec="seconds"),
        "results": results,
        "n_jobs": len(jobs),
    }
    save_health(report)
    return report


def describe_registry() -> pd.DataFrame:
    """把注册表整理成一张表，供 ``eq data sources`` 展示。"""
    rows = []
    for s in sorted(REGISTRY.values(), key=lambda x: (x.priority, x.name)):
        rows.append({
            "源": s.name, "名称": s.label,
            "市场": "/".join(sorted(s.markets)),
            "能力": "/".join(s.caps),
            "优先级": s.priority,
            "依赖": ",".join(s.requires) or "-",
            "已装": "是" if s.installed() else "否",
            "需key": "是" if s.needs_key else "否",
            "说明": s.note,
        })
    return pd.DataFrame(rows)
