"""全市场扫描：按涨幅/成交量/成交额排序展示前 N 名。

数据源策略（已验证）：
- A股（.SH/.SZ/.BJ）：akshare 新浪源 `stock_zh_a_spot()`，9s 拉全量 5500+ 只 ✅
- 港股（.HK）：akshare 东财知名港股 `stock_hk_famous_spot_em()`，2s 拉 100 只 ✅
- 美股（.US）：akshare 东财知名美股 `stock_us_famous_spot_em()`，2s 拉 29 只 ✅
- 加密（USDT/）：ccxt OKX `fetch_tickers()`，即时拉 1200+ 交易对 ✅

代码格式转换：akshare 新浪用 `sz302132` / `sh600519` / `bj920000`，EternityQuant 用 `302132.SZ` / `600519.SH` / `920000.BJ`。
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

SortBy = Literal["change_pct", "volume", "amount"]
Market = Literal["A", "HK", "US", "CRYPTO"]

# ---------- A 股 ----------

_A_PREFIX_TO_SUFFIX = {"sh": "SH", "sz": "SZ", "bj": "BJ"}


def _akshare_code_to_eq(code: str) -> str:
    """`sz302132` → `302132.SZ`。"""
    prefix = code[:2].lower()
    rest = code[2:]
    suffix = _A_PREFIX_TO_SUFFIX.get(prefix)
    if suffix is None:
        return code
    return f"{rest}.{suffix}"


def _norm_cols(df: pd.DataFrame, col_map: dict[str, str], sort_by: SortBy, top_n: int) -> pd.DataFrame:
    """统一列名 + 排序 + 截取前 N。

    上游 akshare 接口的列名时不时变（或某市场压根没有成交额列），
    此前用 ``df[list(col_map.values())]`` 硬取会直接 KeyError 整个命令挂掉；
    改成只取实际存在的列。
    """
    df = df.rename(columns=col_map)
    keep = [c for c in col_map.values() if c in df.columns]
    if "symbol" not in keep:
        raise ValueError(f"数据源返回的列不含代码列，实际列：{list(df.columns)[:12]}")
    df = df[keep].copy()
    for col in ["close", "change_pct", "volume", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    subset = [c for c in ("close", "change_pct") if c in df.columns]
    if subset:
        df = df.dropna(subset=subset)
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    elif "change_pct" in df.columns:
        # 该市场没有这个排序键（如美股知名榜无成交量列）时退回涨跌幅，
        # 而不是"不排序且不截断"——此前会把全表都返回。
        df = df.sort_values("change_pct", ascending=False)
    return df.head(top_n).reset_index(drop=True)


def scan_a_share(sort_by: SortBy = "change_pct", top_n: int = 30) -> pd.DataFrame:
    """扫 A 股全市场（新浪源，5500+ 只 9s）。"""
    import akshare as ak
    raw = ak.stock_zh_a_spot()
    col_map = {
        "代码": "symbol", "名称": "name", "最新价": "close",
        "涨跌幅": "change_pct", "成交量": "volume", "成交额": "amount",
        "今开": "open", "最高": "high", "最低": "low", "昨收": "prev_close",
    }
    df = _norm_cols(raw, col_map, sort_by, top_n)
    df["symbol"] = df["symbol"].map(_akshare_code_to_eq)
    return df


# ---------- 港股 ----------

def scan_hk(sort_by: SortBy = "change_pct", top_n: int = 30) -> pd.DataFrame:
    """扫港股知名列表（东财源，100 只 2s）。"""
    import akshare as ak
    raw = ak.stock_hk_famous_spot_em()
    col_map = {
        "代码": "symbol", "名称": "name", "最新价": "close",
        "涨跌幅": "change_pct", "成交量": "volume", "成交额": "amount",
        "今开": "open", "最高": "high", "最低": "low", "昨收": "prev_close",
    }
    df = _norm_cols(raw, col_map, sort_by, top_n)
    # 代码格式：00981 → 00981.HK（东财港股代码无后缀）
    df["symbol"] = df["symbol"].astype(str).str.zfill(5) + ".HK"
    return df


# ---------- 美股 ----------

def scan_us(sort_by: SortBy = "change_pct", top_n: int = 30) -> pd.DataFrame:
    """扫美股知名列表（东财源，29 只 2s）。"""
    import akshare as ak
    raw = ak.stock_us_famous_spot_em()
    col_map = {
        "代码": "symbol", "名称": "name", "最新价": "close",
        "涨跌幅": "change_pct", "开盘价": "open", "最高价": "high",
        "最低价": "low", "昨收价": "prev_close",
    }
    df = _norm_cols(raw, col_map, sort_by, top_n)
    # 代码格式：105.NVDA → NVDA.US；东财偶尔回不带交易所前缀的裸代码（NVDA），
    # 此前无脑取 split(".")[1] 会得到 NaN，整行符号变成 nan → 后续全炸。
    codes = df["symbol"].astype(str).str.rsplit(".", n=1).str[-1]
    df["symbol"] = codes.str.upper() + ".US"
    return df


# ---------- 加密 ----------

def scan_crypto(sort_by: SortBy = "change_pct", top_n: int = 30) -> pd.DataFrame:
    """扫加密市场（ccxt OKX，1200+ 交易对实时）。"""
    import ccxt
    okx = ccxt.okx()
    tickers = okx.fetch_tickers()
    rows = []
    for sym, t in tickers.items():
        if not sym.endswith("/USDT"):
            continue
        pct = t.get("percentage")
        if pct is None:
            continue
        rows.append({
            "symbol": sym.replace("/", "-"),
            "name": sym.replace("/USDT", ""),
            "close": t.get("last") or 0,
            "change_pct": pct,
            "volume": t.get("baseVolume") or 0,
            "amount": (t.get("last") or 0) * (t.get("baseVolume") or 0),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    sort_key = {"change_pct": "change_pct", "volume": "volume", "amount": "amount"}.get(sort_by, "change_pct")
    df = df.sort_values(sort_key, ascending=False).head(top_n).reset_index(drop=True)
    return df


# ---------- 统一入口 ----------

_SCANNERS = {
    "A": scan_a_share,
    "HK": scan_hk,
    "US": scan_us,
    "CRYPTO": scan_crypto,
}


def scan(market: Market, sort_by: SortBy = "change_pct", top_n: int = 30,
         prefer: list[str] | None = None) -> pd.DataFrame:
    """统一入口：按市场扫描全市场快照。

    v0.26 起先走 :mod:`eq.data.sources` 注册表（东财/Binance/OKX/akshare 依次
    failover），注册表全挂再退回原来写死的 akshare 扫描器。
    """
    if market not in _SCANNERS:
        raise ValueError(f"未知市场 {market}，可选：{list(_SCANNERS)}")

    from eq.data import sources as src_reg

    try:
        df, used = src_reg.fetch_spot(market, top_n=max(top_n * 3, 100), prefer=prefer)
        logger.debug("%s 全市场快照取自 %s", market, used)
        sort_key = sort_by if sort_by in df.columns else "change_pct"
        if sort_key in df.columns:
            df = df.sort_values(sort_key, ascending=False)
        return df.head(top_n).reset_index(drop=True)
    except Exception as e:
        logger.warning("数据源注册表扫描 %s 失败（%s），退回 akshare 扫描器", market, e)

    return _SCANNERS[market](sort_by=sort_by, top_n=top_n)


# ---------- 格式化输出 ----------

_MARKET_LABELS = {"A": "A 股", "HK": "港股", "US": "美股", "CRYPTO": "加密"}

_SORT_LABELS = {"change_pct": "涨跌幅", "volume": "成交量", "amount": "成交额"}


def use_color() -> bool:
    """终端是否该上色。重定向到文件 / 设了 NO_COLOR 时关掉，避免报表里混入转义序列。"""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("EQ_FORCE_COLOR"):
        return True
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def format_scan(df: pd.DataFrame, sort_by: SortBy, market: Market = "A", color: bool | None = None) -> str:
    """格式化扫描结果为文本表格。"""
    sort_label = _SORT_LABELS.get(sort_by, sort_by)
    market_label = _MARKET_LABELS.get(market, market)
    colorize = use_color() if color is None else color
    lines = [f"\n{market_label} 按 {sort_label} 排序，前 {len(df)} 名：\n"]
    # 名称列宽此前 header 写 12、数据行写 10，整张表右半边错位
    header = f"{'代码':<14} {'名称':<12} {'最新价':>10} {'涨跌幅':>10} {'成交量':>14} {'成交额':>14}"
    lines.append(header)
    lines.append("-" * 88)
    for _, row in df.iterrows():
        pct = float(row.get("change_pct", 0) or 0)
        arrow = "▲" if pct >= 0 else "▼"
        color_on = ("\033[91m" if pct >= 0 else "\033[92m") if colorize else ""
        reset = "\033[0m" if colorize else ""
        name = str(row.get("name", ""))[:12]
        close = float(row.get("close", 0) or 0)
        volume = float(row.get("volume", 0) or 0)
        amount = float(row.get("amount", 0) or 0)
        lines.append(
            f"{row['symbol']:<14} {name:<12} {close:>10.2f} "
            f"{color_on}{arrow}{pct:+8.2f}%{reset} "
            f"{volume:>14.0f} {amount:>14.0f}"
        )
    return "\n".join(lines) + "\n"