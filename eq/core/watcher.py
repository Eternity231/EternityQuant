"""行情快照格式化，供 `eq watch` 命令使用。"""

from __future__ import annotations

from eq.data.market import get_snapshot


def format_snapshot(symbol: str) -> str:
    """拉行情并格式化为多行文本块。"""
    snap = get_snapshot(symbol)
    arrow = "▲" if snap["change_pct"] >= 0 else "▼"
    color = "\033[91m" if snap["change_pct"] >= 0 else "\033[92m"
    reset = "\033[0m"
    return (
        f"\n{snap['symbol']}  {snap['date']}\n"
        f"  开 {snap['open']:<10.2f}  高 {snap['high']:<10.2f}  低 {snap['low']:<10.2f}\n"
        f"  收 {snap['close']:<10.2f}  量 {snap['volume']:<14.0f}\n"
        f"  前收 {snap['prev_close']:<10.2f}  涨跌 {color}{arrow} {snap['change_pct']:+.2f}%{reset}\n"
    )
