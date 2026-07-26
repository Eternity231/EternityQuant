"""市场数据获取：按市场选主源 + fallback。

直调 yfinance / akshare / baostock SDK，不经任何外部服务或 AI agent。
"""

from __future__ import annotations

import atexit
import datetime as dt
import logging
import re
import threading
from contextlib import contextmanager
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

Market = Literal["A", "HK", "US", "CRYPTO"]

# A 股代码识别：6 位数字 + .SH/.SZ/.BJ
_A_SHARE_RE = re.compile(r"^[0-9]{6}\.(SH|SZ|BJ)$")
# 港股：5 位数字 + .HK（部分 4 位）
_HK_RE = re.compile(r"^[0-9]{4,5}\.HK$")
# 美股：字母代码 + .US
_US_RE = re.compile(r"^[A-Z.]+\.(US|NY|NQ)$")
# 加密：BTC-USDT / ETH-USDT 形式
_CRYPTO_RE = re.compile(r"^[A-Z0-9]+-[A-Z0-9]+$")

# 裸代码（无后缀）识别，用于 normalize_symbol
_BARE_A_RE = re.compile(r"^[0-9]{6}$")
# 港股裸码 1~5 位（A 股裸码固定 6 位，所以 ≤5 位纯数字不会歧义）
_BARE_HK_RE = re.compile(r"^[0-9]{1,5}$")
_BARE_US_RE = re.compile(r"^[A-Z][A-Z.\-]{0,6}$")


def normalize_symbol(symbol: str) -> str:
    """把用户随手输入的符号规整成项目标准格式。

    容错处理（此前这些输入一律 ``ValueError: 无法识别市场``）：

    - 大小写与首尾空白：``600519.sh`` → ``600519.SH``
    - 裸 A 股 6 位码按板块补后缀：``600519`` → ``600519.SH``，``000001`` → ``000001.SZ``
    - 裸港股 4/5 位码补零补后缀：``700`` → ``00700.HK``
    - 港股 4 位补零：``0700.HK`` → ``00700.HK``
    - 裸美股字母码：``AAPL`` → ``AAPL.US``
    - qlib 风格前缀：``SH600519`` → ``600519.SH``
    """
    s = str(symbol).strip().strip('"').strip("'").upper()
    if not s:
        raise ValueError("符号为空")
    # 加密对（BTC-USDT）原样返回
    if _CRYPTO_RE.match(s):
        return s
    # qlib 风格前缀 SH600519 / SZ000001 / BJ920000
    if len(s) == 8 and s[:2] in ("SH", "SZ", "BJ") and s[2:].isdigit():
        return f"{s[2:]}.{s[:2]}"
    if "." in s:
        code, _, suffix = s.rpartition(".")
        if suffix == "HK" and code.isdigit():
            return f"{code.zfill(5)}.HK"
        return s
    if _BARE_A_RE.match(s):
        # 北交所要先判：920xxx 是北交所新代码段，若先走 "9→沪" 会误判成上交所
        # （上交所以 9 开头的是 900xxx B 股）
        if s.startswith("92") or s[0] in "48":
            return f"{s}.BJ"
        if s[0] in "69":
            return f"{s}.SH"
        if s[0] in "03":
            return f"{s}.SZ"
        return f"{s}.SH"
    if _BARE_HK_RE.match(s):
        return f"{s.zfill(5)}.HK"
    if _BARE_US_RE.match(s):
        return f"{s}.US"
    return s


def detect_market(symbol: str) -> Market:
    """根据符号格式识别市场。输入会先经 :func:`normalize_symbol` 规整。"""
    s = normalize_symbol(symbol)
    if _A_SHARE_RE.match(s):
        return "A"
    if _HK_RE.match(s):
        return "HK"
    if _US_RE.match(s):
        return "US"
    if _CRYPTO_RE.match(s):
        return "CRYPTO"
    raise ValueError(f"无法识别市场：{symbol}")


def yfinance_symbol(symbol: str, market: Market) -> str:
    """把 EternityQuant 符号转成 yfinance 符号。"""
    code, _, suffix = symbol.partition(".")
    if market == "A":
        # yfinance 用 600519.SS / 000001.SZ
        return f"{code}.{('SS' if suffix == 'SH' else 'SZ') if suffix in ('SH', 'SZ') else 'BJ'}"
    if market == "HK":
        # yfinance 港股是 4 位零填充（0700.HK），项目内标准是 5 位（00700.HK）。
        # 直接传 5 位 yfinance 查无此票 → 之前港股主源必然失败再退 akshare。
        return f"{code.lstrip('0').zfill(4)}.HK"
    if market == "US":
        return code  # yfinance 美股不带后缀
    # 加密：项目用 BTC-USDT，yfinance 只认法币计价的 BTC-USD
    base, _, quote = symbol.partition("-")
    return f"{base}-{'USD' if quote in ('USDT', 'USDC', 'BUSD') else quote}"


def _akshare_symbol(symbol: str, market: Market) -> str:
    """akshare 调用所需的符号（akshare 接口各异，后续按接口分）。"""
    return symbol


def bare_code(symbol: str) -> str:
    """剥掉市场后缀，拿 akshare/东财这类接口要的裸代码。``600519.SH`` → ``600519``。"""
    return normalize_symbol(symbol).partition(".")[0]


def get_recent_bars(
    symbol: str,
    days: int = 30,
    *,
    use_cache: bool = True,
    ttl_seconds: int | None = None,
    prefer: list[str] | None = None,
) -> pd.DataFrame:
    """拉取最近 N 个交易日的日线 OHLCV。

    v0.26 起走 :mod:`eq.data.sources` 注册表：按「本机自检可用 → priority」
    的顺序逐个源尝试，某个源挂了自动换下一个，不再是写死的两级链。
    用 ``eq data sources --test`` 可以在你自己的机器上实测各源可用性。

    Args:
        days: 需要的交易日根数（返回值最多 ``days`` 行，取最近的）
        use_cache: 是否读写本地缓存（``eq`` 的 ``--no-cache`` 会关掉）
        ttl_seconds: 缓存新鲜度阈值，缺省 6 小时
        prefer: 强制优先用某几个源，如 ``["tencent", "sina"]``

    Returns:
        DataFrame indexed by date, columns: open/high/low/close/volume
    """
    symbol = normalize_symbol(symbol)
    market = detect_market(symbol)
    end = dt.date.today()
    start = end - dt.timedelta(days=max(days * 2, days + 10))  # 留出非交易日冗余

    if use_cache:
        from eq.data import cache as bar_cache

        ttl = bar_cache.DEFAULT_TTL_SECONDS if ttl_seconds is None else ttl_seconds
        if bar_cache.is_fresh(symbol, start, end, ttl_seconds=ttl):
            cached = bar_cache.load_bars(symbol, start, end)
            if not cached.empty:
                logger.debug("缓存命中 %s（%d 行）", symbol, len(cached))
                return cached.tail(days)

    from eq.data import sources as src_reg

    try:
        df, used = src_reg.fetch_bars(symbol, market, start, end, prefer=prefer)
        logger.debug("%s 日线取自 %s", symbol, used)
    except src_reg.AllSourcesFailed:
        # 全部源都挂时退化到缓存（哪怕过期），总比直接报错强
        if use_cache:
            from eq.data import cache as bar_cache

            stale = bar_cache.load_bars(symbol, start, end)
            if not stale.empty:
                logger.warning("%s 全部数据源失败，退化用过期缓存（%d 行）", symbol, len(stale))
                return stale.tail(days)
        raise

    if use_cache and not df.empty:
        from eq.data import cache as bar_cache

        bar_cache.save_bars(symbol, df)
    return df.tail(days)


def _baostock_symbol(symbol: str) -> str:
    """把 EternityQuant A 股符号转成 baostock 符号（sh.600519 / sz.000001）。"""
    code, _, suffix = symbol.partition(".")
    prefix = {"SH": "sh", "SZ": "sz", "BJ": "bj"}.get(suffix, suffix.lower())
    return f"{prefix}.{code}"


# baostock 用的是**进程级全局 socket**，多线程并发调用会互相踩踏
# （典型症状：WinError 10038 "在一个非套接字上尝试了一个操作"）。
# get_snapshots / screener 都是线程池并发，所以这里必须串行化。
# 同时把 login 提到进程级复用：此前每只票 login+logout 一次，
# 批量拉 50 只就是 50 次握手。
_BS_LOCK = threading.RLock()
_bs_logged_in = False


def _bs_logout() -> None:
    global _bs_logged_in
    if _bs_logged_in:
        try:
            import baostock as bs

            bs.logout()
        except Exception:
            pass
        _bs_logged_in = False


@contextmanager
def _baostock_session():
    """串行化 + 复用 baostock 登录态。整个 with 块内独占 baostock。"""
    global _bs_logged_in
    with _BS_LOCK:
        import baostock as bs  # 延迟加载

        if not _bs_logged_in:
            lg = bs.login()
            if lg.error_code != "0":
                raise ValueError(f"baostock login 失败：{lg.error_msg}")
            _bs_logged_in = True
            atexit.register(_bs_logout)
        try:
            yield bs
        except Exception:
            # 查询出错时登录态可能已损坏，下次重新登录
            _bs_logged_in = False
            try:
                bs.logout()
            except Exception:
                pass
            raise


def _fetch_baostock_a(symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """baostock 拉取 A 股日线。TCP 直连，不依赖 HTTP 爬虫，无 IP 限流。"""
    bs_code = _baostock_symbol(symbol)
    with _baostock_session() as bs:
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume",
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
            frequency="d",
            adjustflag="2",  # 前复权
        )
        if rs.error_code != "0":
            raise ValueError(f"baostock 查询失败：{rs.error_msg}")
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
    if not rows:
        raise ValueError(f"baostock 返回空：{symbol}")
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _fetch_yfinance(symbol: str, market: Market, start: dt.date, end: dt.date) -> pd.DataFrame:
    import yfinance as yf  # 延迟加载，避免未安装时阻塞 CLI

    yf_symbol = yfinance_symbol(symbol, market)
    df = yf.download(yf_symbol, start=start.isoformat(), end=end.isoformat(), progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"yfinance 返回空：{yf_symbol}")
    # yfinance 返回 MultiIndex 列（symbol 一级），扁平化
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    return df.dropna()


def _fetch_akshare_a(symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    import akshare as ak  # 延迟加载

    code = symbol.partition(".")[0]
    # akshare A 股接口：stock_zh_a_hist，符号格式 600519（不带后缀）
    df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"), adjust="qfq")
    if df is None or df.empty:
        raise ValueError(f"akshare 返回空：{symbol}")
    df = df.rename(columns={"开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"})
    df = df.set_index("日期")
    df.index = pd.to_datetime(df.index)
    return df[["open", "high", "low", "close", "volume"]]


def _fetch_akshare_fallback(symbol: str, market: Market, start: dt.date, end: dt.date) -> pd.DataFrame:
    """akshare 作为兜底源。A/HK/US fallback。"""
    if market == "A":
        return _fetch_akshare_a(symbol, start, end)
    if market == "HK":
        import akshare as ak
        code, _, _ = symbol.partition(".")
        # akshare 港股：stock_hk_hist(symbol=...)，不是 symbol_em
        df = ak.stock_hk_hist(symbol=code, period="daily", start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"), adjust="qfq")
        df = df.rename(columns={"开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"})
        df = df.set_index("日期")
        df.index = pd.to_datetime(df.index)
        return df[["open", "high", "low", "close", "volume"]]
    if market == "US":
        import akshare as ak
        code, _, _ = symbol.partition(".")
        # akshare 美股：stock_us_hist(symbol=..., adjust='qfq')
        df = ak.stock_us_hist(symbol=code, period="daily", start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"), adjust="qfq")
        df = df.rename(columns={"开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"})
        df = df.set_index("日期")
        df.index = pd.to_datetime(df.index)
        return df[["open", "high", "low", "close", "volume"]]
    raise NotImplementedError(f"akshare fallback for {market} 待集成")


def get_snapshot(
    symbol: str,
    *,
    use_cache: bool = True,
    realtime: bool = False,
    prefer: list[str] | None = None,
) -> dict[str, float | str]:
    """拉最近一根日线 + 前一日对比，返回快照字典。

    用于 `eq watch` 命令的显示。

    Args:
        realtime: True 时优先走**实时行情源**（新浪/腾讯，盘中返回当前价而非
            上一交易日收盘价），失败再退回日线推导。默认 False——日线推导的
            结果和回测/训练用的数据同源，口径一致、可缓存。
        prefer: 强制优先用某几个源
    """
    symbol = normalize_symbol(symbol)

    if realtime:
        from eq.data import sources as src_reg

        try:
            snap, used = src_reg.fetch_snapshot(symbol, detect_market(symbol), prefer=prefer)
            logger.debug("%s 实时快照取自 %s", symbol, used)
            snap.setdefault("source", used)
            return snap
        except Exception as e:
            logger.debug("实时快照失败（%s），退回日线推导：%s", symbol, e)

    df = get_recent_bars(symbol, days=5, use_cache=use_cache, prefer=prefer)
    if df.empty:
        raise ValueError(f"无数据：{symbol}")
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last
    close = float(last["close"])
    prev_close = float(prev["close"])
    change_pct = (close - prev_close) / prev_close * 100 if prev_close else 0.0
    return {
        "symbol": symbol,
        "date": str(df.index[-1].date()),
        "open": float(last["open"]),
        "high": float(last["high"]),
        "low": float(last["low"]),
        "close": close,
        "volume": float(last["volume"]),
        "prev_close": prev_close,
        "change_pct": change_pct,
    }


def get_snapshots(
    symbols: list[str], *, use_cache: bool = True, workers: int = 8,
    realtime: bool = False, prefer: list[str] | None = None,
) -> dict[str, dict[str, float | str] | None]:
    """并发拉一批标的的快照。拉不到的键值为 ``None``，不抛异常。

    自选股/持仓这类「一屏几十只」的场景，串行拉是几十秒起步；
    这里用线程池并发（数据源都是 IO 阻塞，GIL 不是瓶颈）。
    """
    from concurrent.futures import ThreadPoolExecutor

    symbols = [normalize_symbol(s) for s in symbols]
    if not symbols:
        return {}

    # 实时模式下优先走**批量**接口：新浪/腾讯支持一次请求问几十只，
    # 50 只自选从 50 次网络往返压成 1 次。
    if realtime:
        from eq.data import sources as src_reg

        by_market: dict[str, list[str]] = {}
        for s in symbols:
            try:
                by_market.setdefault(detect_market(s), []).append(s)
            except ValueError:
                logger.debug("跳过无法识别市场的符号：%s", s)
        out: dict[str, dict[str, float | str] | None] = dict.fromkeys(symbols)
        leftover: list[str] = []
        for mkt, syms in by_market.items():
            try:
                got, used = src_reg.fetch_batch(syms, mkt, prefer=prefer)
                for sym in syms:
                    snap = got.get(sym)
                    if snap is None:
                        leftover.append(sym)
                    else:
                        snap.setdefault("source", used)
                        out[sym] = snap
            except Exception as e:
                logger.debug("%s 批量快照失败（%s），逐只回退", mkt, e)
                leftover.extend(syms)
        if not leftover:
            return out
        # 批量没覆盖到的少数标的再逐只补
        symbols = leftover
    else:
        out = {}

    def _one(sym: str):
        try:
            return sym, get_snapshot(sym, use_cache=use_cache,
                                     realtime=realtime, prefer=prefer)
        except Exception as e:
            logger.debug("快照失败 %s：%s", sym, e)
            return sym, None

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(symbols)))) as pool:
        out.update(dict(pool.map(_one, symbols)))
    return out
