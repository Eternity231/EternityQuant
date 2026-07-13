"""市场数据获取：按市场选主源 + fallback。

vibe-trading 仅作数据补全和一次性查询的助手，不作硬依赖。
本模块直调 yfinance / akshare SDK，避免 MCP stdio RPC 开销。
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

Market = Literal["A", "HK", "US", "CRYPTO"]

# A 股代码识别：6 位数字 + .SH/.SZ/.BJ
_A_SHARE_RE = re.compile(r"^[0-9]{6}\.(SH|SZ|BJ)$")
# 港股：5 位数字 + .HK（部分 4 位）
_HK_RE = re.compile(r"^[0-9]{4,5}\.HK$")
# 美股：字母代码 + .US
_US_RE = re.compile(r"^[A-Z]+\.(US|NY|NQ)$")
# 加密：BTC-USDT / ETH-USDT 形式
_CRYPTO_RE = re.compile(r"^[A-Z]+-[A-Z]+$")


def detect_market(symbol: str) -> Market:
    """根据符号格式识别市场。"""
    if _A_SHARE_RE.match(symbol):
        return "A"
    if _HK_RE.match(symbol):
        return "HK"
    if _US_RE.match(symbol):
        return "US"
    if _CRYPTO_RE.match(symbol):
        return "CRYPTO"
    raise ValueError(f"无法识别市场：{symbol}")


def _yfinance_symbol(symbol: str, market: Market) -> str:
    """把 EternityQuant 符号转成 yfinance 符号。"""
    code, _, suffix = symbol.partition(".")
    if market == "A":
        # yfinance 用 600519.SS / 000001.SZ
        return f"{code}.{('SS' if suffix == 'SH' else 'SZ') if suffix in ('SH', 'SZ') else 'BJ'}"
    if market == "HK":
        return f"{code}.HK"  # yfinance 港股用零 padding 不一致，直接试
    if market == "US":
        return code  # yfinance 美股不带后缀
    return symbol.replace("-", "-")  # 加密保持 BTC-USDT


def _akshare_symbol(symbol: str, market: Market) -> str:
    """akshare 调用所需的符号（akshare 接口各异，后续按接口分）。"""
    return symbol


def get_recent_bars(symbol: str, days: int = 30) -> pd.DataFrame:
    """拉取最近 N 个交易日的日线 OHLCV。

    按市场选主源，失败切 akshare fallback。

    Returns:
        DataFrame indexed by date, columns: open/high/low/close/volume
    """
    market = detect_market(symbol)
    end = dt.date.today()
    start = end - dt.timedelta(days=days * 2)  # 留出非交易日冗余

    # 主源：A 股 → baostock（TCP 直连稳定）；港股/美股/加密 → yfinance
    try:
        if market == "A":
            return _fetch_baostock_a(symbol, start, end)
        return _fetch_yfinance(symbol, market, start, end)
    except Exception as e:
        logger.warning("主源拉取 %s 失败：%s，切 akshare fallback", symbol, e)
        return _fetch_akshare_fallback(symbol, market, start, end)


def _baostock_symbol(symbol: str) -> str:
    """把 EternityQuant A 股符号转成 baostock 符号（sh.600519 / sz.000001）。"""
    code, _, suffix = symbol.partition(".")
    prefix = {"SH": "sh", "SZ": "sz", "BJ": "bj"}.get(suffix, suffix.lower())
    return f"{prefix}.{code}"


def _fetch_baostock_a(symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """baostock 拉取 A 股日线。TCP 直连，不依赖 HTTP 爬虫，无 IP 限流。"""
    import baostock as bs  # 延迟加载

    bs_code = _baostock_symbol(symbol)
    lg = bs.login()
    if lg.error_code != "0":
        raise ValueError(f"baostock login 失败：{lg.error_msg}")
    try:
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
    finally:
        bs.logout()


def _fetch_yfinance(symbol: str, market: Market, start: dt.date, end: dt.date) -> pd.DataFrame:
    import yfinance as yf  # 延迟加载，避免未安装时阻塞 CLI

    yf_symbol = _yfinance_symbol(symbol, market)
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

    code, _, suffix = symbol.partition(".")
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


def get_snapshot(symbol: str) -> dict[str, float | str]:
    """拉最近一根日线 + 前一日对比，返回快照字典。

    用于 `eq watch` 命令的显示。
    """
    df = get_recent_bars(symbol, days=5)
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
