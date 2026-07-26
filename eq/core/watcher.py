"""行情快照格式化，供 `eq watch` 命令使用。"""

from __future__ import annotations

from eq.data.market import get_snapshot


def format_snapshot(symbol: str, *, realtime: bool = False,
                    prefer: list[str] | None = None) -> str:
    """拉行情并格式化为多行文本块。"""
    snap = get_snapshot(symbol, realtime=realtime, prefer=prefer)
    arrow = "▲" if snap["change_pct"] >= 0 else "▼"
    color = "\033[91m" if snap["change_pct"] >= 0 else "\033[92m"
    reset = "\033[0m"
    # 实时源（新浪/腾讯）会带股票名和源名，日线推导的路径没有
    head = f"\n{snap['symbol']}"
    if snap.get("name"):
        head += f"  {snap['name']}"
    head += f"  {snap['date']}\n"
    tail = ""
    if snap.get("source"):
        tail = f"  数据源 {snap['source']}（实时）\n"
    return (
        head
        + f"  开 {snap['open']:<10.2f}  高 {snap['high']:<10.2f}  低 {snap['low']:<10.2f}\n"
        + f"  收 {snap['close']:<10.2f}  量 {snap['volume']:<14.0f}\n"
        + f"  前收 {snap['prev_close']:<10.2f}  涨跌 {color}{arrow} {snap['change_pct']:+.2f}%{reset}\n"
        + tail
    )
