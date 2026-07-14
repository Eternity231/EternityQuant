"""港股数据工具（akshare Sina 源，大陆网络下唯一可用的港股数据源）。

Sina 源港股快照（stock_hk_spot）2799 只约 83 秒。
Sina 源港股日线（stock_hk_hist）单只约 2-5 秒。
东财/腾讯源在大陆被限流（RemoteDisconnected）。
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd

_HK_DATA_DIR = Path(__file__).resolve().parent.parent.parent / ".eternityquant" / "hk_data"


def _ensure_dir() -> Path:
    _HK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return _HK_DATA_DIR


def list_hk_stocks(limit: int = 100) -> list[str]:
    """拉热门港股列表（新浪源，全量 2799 只约 83s，截取前 limit 只）。"""
    import akshare as ak
    try:
        df = ak.stock_hk_spot()
        if df.empty:
            return []
        # 取前 limit 只的代码
        if "代码" in df.columns:
            codes = df["代码"].astype(str).str.zfill(5).tolist()[:limit]
        else:
            codes = df.iloc[:, 0].astype(str).str.zfill(5).tolist()[:limit]
        return codes
    except Exception:
        return []


def fetch_hk_hist(symbol: str, start: str, end: str) -> pd.DataFrame:
    """拉单只港股日线（新浪源）。symbol：00700（纯代码，无后缀）。"""
    import akshare as ak
    try:
        df = ak.stock_hk_hist(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="qfq")
        if df.empty:
            return pd.DataFrame()
        col_map = {"开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"}
        df = df.rename(columns=col_map)
        df = df.set_index("日期")
        df.index = pd.to_datetime(df.index)
        return df[["open", "high", "low", "close", "volume"]]
    except Exception:
        return pd.DataFrame()


def compute_features_hk(df: pd.DataFrame) -> pd.DataFrame:
    """为港股计算基础技术指标，作为 Alpha158 的替代（港股无法用 qlib Alpha158）。"""
    df = df.copy()
    # 简单移动均线
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    # 涨跌幅
    df["ret1"] = df["close"].pct_change(1)
    df["ret5"] = df["close"].pct_change(5)
    df["ret10"] = df["close"].pct_change(10)
    # 波动率
    df["volatility5"] = df["ret1"].rolling(5).std()
    df["volatility10"] = df["ret1"].rolling(10).std()
    # 成交量
    df["volume_ma5"] = df["volume"].rolling(5).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma5"]
    # 高低价差
    df["high_low_ratio"] = (df["high"] - df["low"]) / df["close"]
    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi14"] = 100 - 100 / (1 + gain / (loss + 1e-10))
    return df.dropna()
