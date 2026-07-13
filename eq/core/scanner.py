"""全市场扫描：按涨幅/成交量/成交额排序展示前 N 名。

第一版只支持 A 股（akshare 新浪源稳定，东财源被拒）。港股/美股/加密待集成。
代码格式转换：akshare 新浪用 `sz302132` / `sh600519` / `bj920000`，EternityQuant 用 `302132.SZ` / `600519.SH` / `920000.BJ`。
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

SortBy = Literal["change_pct", "volume", "amount"]


_A_PREFIX_TO_SUFFIX = {"sh": "SH", "sz": "SZ", "bj": "BJ"}


def _akshare_code_to_eq(code: str) -> str:
    """`sz302132` → `302132.SZ`。"""
    prefix = code[:2].lower()
    rest = code[2:]
    suffix = _A_PREFIX_TO_SUFFIX.get(prefix)
    if suffix is None:
        return code  # 不认识的格式原样返回
    return f"{rest}.{suffix}"


def scan_a_share(sort_by: SortBy = "change_pct", top_n: int = 30) -> pd.DataFrame:
    """扫 A 股全市场，按指定字段降序排，返回前 N 名。

    Returns:
        DataFrame columns: symbol, name, close, change_pct, volume, amount, open, high, low, prev_close
    """
    import akshare as ak  # 延迟加载

    raw = ak.stock_zh_a_spot()  # 新浪源，稳
    col_map = {
        "代码": "symbol",
        "名称": "name",
        "最新价": "close",
        "涨跌幅": "change_pct",
        "成交量": "volume",
        "成交额": "amount",
        "今开": "open",
        "最高": "high",
        "最低": "low",
        "昨收": "prev_close",
    }
    df = raw.rename(columns=col_map)[list(col_map.values())]
    df["symbol"] = df["symbol"].map(_akshare_code_to_eq)
    # 排序键有 0 或 NaN 的去掉（停牌等）
    for col in ["close", "change_pct", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close", "change_pct"])
    df = df.sort_values(sort_by, ascending=False).head(top_n).reset_index(drop=True)
    return df


def format_scan(df: pd.DataFrame, sort_by: SortBy) -> str:
    """格式化扫描结果为文本表格。"""
    sort_label = {"change_pct": "涨跌幅", "volume": "成交量", "amount": "成交额"}[sort_by]
    lines = [f"\n按 {sort_label} 排序，前 {len(df)} 名：\n"]
    header = f"{'代码':<12} {'名称':<10} {'最新价':>10} {'涨跌幅':>10} {'成交量':>14} {'成交额':>14}"
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in df.iterrows():
        arrow = "▲" if row["change_pct"] >= 0 else "▼"
        color = "\033[91m" if row["change_pct"] >= 0 else "\033[92m"
        reset = "\033[0m"
        name = str(row["name"])[:8]  # 中文字符占 2 字符宽，截 4 个汉字
        lines.append(
            f"{row['symbol']:<12} {name:<8} {row['close']:>10.2f} "
            f"{color}{arrow}{row['change_pct']:+7.2f}%{reset} "
            f"{row['volume']:>14.0f} {row['amount']:>14.0f}"
        )
    return "\n".join(lines) + "\n"
