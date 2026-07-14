"""统一数据收集器 — 支持 A 股/港股/美股/分钟线。

用法：
    python -m eq.data.collector a            # A 股日线（qlib/baostock）
    python -m eq.data.collector hk           # 港股日线（yfinance）
    python -m eq.data.collector hk_5min      # 港股 5 分钟线（yfinance，最近 60 天）
    python -m eq.data.collector hk_1min      # 港股 1 分钟线（yfinance，最近 7 天）
    python -m eq.data.collector us           # 美股日线（yfinance）
    python -m eq.data.collector all          # 全部
"""

from __future__ import annotations

import datetime as dt
import os
import time
from pathlib import Path

import pandas as pd

# 缓存目录
_HK_DIR = Path(__file__).resolve().parent.parent.parent / ".eternityquant" / "hk_data"
_US_DIR = Path(__file__).resolve().parent.parent.parent / ".eternityquant" / "us_data"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _fmt_yf_hk(code: str) -> str:
    """港股代码转 yfinance 格式：09988 → 9988.HK，00700 → 0700.HK"""
    c = code.lstrip("0")
    while len(c) < 4:
        c = "0" + c
    return c + ".HK"


def collect_hk_daily(
    codes: list[str] | None = None,
    top_n: int = 200,
    start: str = "2024-01-01",
    end: str | None = None,
):
    """港股日线（yfinance）。"""
    import yfinance as yf

    if end is None:
        end = dt.date.today().isoformat()
    if codes is None:
        # 默认热门港股列表
        codes = [
            "00700", "09988", "01024", "01810", "09626", "09888", "09999",
            "03690", "01211", "02015", "02318", "02628", "01299", "00005",
            "00011", "00388", "00883", "00941", "00981", "01347",
        ]

    out = _HK_DIR / "daily"
    _ensure_dir(out)
    ok = 0
    for code in codes[:top_n]:
        path = out / f"{code}.csv"
        if path.exists() and path.stat().st_size > 1000:
            ok += 1
            continue
        try:
            yf_code = _fmt_yf_hk(code)
            df = yf.download(yf_code, start=start, end=end, progress=False)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].rename(
                columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
            )
            df.to_csv(path)
            ok += 1
            print(f"  ✓ 港股日线 {code}  {len(df)} 行", flush=True)
        except Exception as e:
            print(f"  ✗ 港股日线 {code}  {str(e)[:50]}", flush=True)
        time.sleep(0.3)
    print(f"  港股日线完成: {ok}/{min(top_n, len(codes))}")


def collect_hk_minute(
    codes: list[str] | None = None,
    top_n: int = 200,
    interval: str = "5m",
    period: str = "2mo",
):
    """港股分钟线（yfinance）。
    
    Args:
        interval: 1m | 5m | 15m | 30m | 60m
        period: 1m=7d, 5m=2mo, 15m/30m/60m=6mo+
    """
    import yfinance as yf

    if codes is None:
        codes = [
            "00700", "09988", "01024", "01810", "09626", "09888", "09999",
            "03690", "01211", "02015", "02318", "02628", "01299", "00005",
            "00011", "00388", "00883", "00941", "00981", "01347",
        ]

    label = f"港股{interval}"
    out = _HK_DIR / interval
    _ensure_dir(out)
    ok = 0
    for code in codes[:top_n]:
        path = out / f"{code}.csv"
        if path.exists() and path.stat().st_size > 1000:
            ok += 1
            continue
        try:
            yf_code = _fmt_yf_hk(code)
            df = yf.download(yf_code, period=period, interval=interval, progress=False)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].rename(
                columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
            )
            df.to_csv(path)
            ok += 1
            print(f"  ✓ {label} {code}  {len(df)} 行", flush=True)
        except Exception as e:
            print(f"  ✗ {label} {code}  {str(e)[:50]}", flush=True)
        time.sleep(0.3)
    print(f"  {label}完成: {ok}/{min(top_n, len(codes))}")


def collect_us_daily(
    codes: list[str] | None = None,
    top_n: int = 100,
    start: str = "2024-01-01",
    end: str | None = None,
):
    """美股日线（yfinance）。"""
    import yfinance as yf

    if end is None:
        end = dt.date.today().isoformat()
    if codes is None:
        codes = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "QQQ", "SPY"]

    out = _US_DIR / "daily"
    _ensure_dir(out)
    ok = 0
    for code in codes[:top_n]:
        path = out / f"{code}.csv"
        if path.exists() and path.stat().st_size > 1000:
            ok += 1
            continue
        try:
            df = yf.download(code, start=start, end=end, progress=False)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].rename(
                columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
            )
            df.to_csv(path)
            ok += 1
            print(f"  ✓ 美股日线 {code}  {len(df)} 行", flush=True)
        except Exception as e:
            print(f"  ✗ 美股日线 {code}  {str(e)[:50]}", flush=True)
        time.sleep(0.3)
    print(f"  美股日线完成: {ok}/{min(top_n, len(codes))}")


def collect_a_share(start: str = "2026-01-01", universe: str = "csi300", workers: int = 10):
    """A 股日线（qlib/baostock）。"""
    from eq.strategy.factors.ml_data_updater import update_qlib_data
    result = update_qlib_data(start=start, universe=universe, workers=workers, verbose=True)
    print(f"  A股({universe})完成: {result}")
    return result


# ===== CLI 入口 =====
def main():
    import sys
    args = sys.argv[1:] if len(sys.argv) > 1 else ["all"]

    for arg in args:
        if arg == "a" or arg == "a_share":
            print("\n=== A 股日线 ===")
            collect_a_share()
        elif arg == "hk" or arg == "hk_daily":
            print("\n=== 港股日线 ===")
            collect_hk_daily()
        elif arg == "hk_5min":
            print("\n=== 港股 5 分钟线 ===")
            collect_hk_minute(interval="5m", period="1mo")
        elif arg == "hk_1min":
            print("\n=== 港股 1 分钟线 ===")
            collect_hk_minute(interval="1m", period="7d")
        elif arg == "us":
            print("\n=== 美股日线 ===")
            collect_us_daily()
        elif arg == "all":
            print("\n===== 全量数据收集 =====\n")
            collect_a_share()
            collect_hk_daily()
            collect_hk_minute(interval="5m", period="2mo")
            collect_hk_minute(interval="1m", period="7d")
            collect_us_daily()
            print("\n===== 全部完成 =====\n")
        else:
            print(f"未知: {arg}，可选: a, hk, hk_5min, hk_1min, us, all")


if __name__ == "__main__":
    main()