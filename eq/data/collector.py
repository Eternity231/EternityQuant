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

from eq.data.paths import (
    HK_DAILY_DIR, HK_5M_DIR, HK_1M_DIR,
    US_DAILY_DIR, ensure_data_dirs,
)


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
    """港股日线（akshare 新浪源，有 VPN 时可用，全历史 2004~2026）。"""
    import akshare as ak

    if end is None:
        end = dt.date.today().isoformat()
    if codes is None:
        codes = [
            "00700", "09988", "01024", "01810", "09626", "09888", "09999",
            "03690", "01211", "02015", "02318", "02628", "01299", "00005",
            "00011", "00388", "00883", "00941", "00981", "01347",
        ]

    out = HK_DAILY_DIR
    ensure_data_dirs()
    ok = 0
    for code in codes[:top_n]:
        path = out / f"{code}.csv"
        if path.exists() and path.stat().st_size > 1000:
            # 检查是否已有足够数据（至少 300 行）
            try:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                if len(df) >= 300:
                    ok += 1
                    continue
            except:
                pass
        try:
            df = ak.stock_hk_daily(symbol=code, adjust="qfq")
            if df.empty:
                continue
            df = df.rename(columns={"date": "Date"}).set_index("Date")
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df = df[["open", "high", "low", "close", "volume"]]
            df.to_csv(path)
            ok += 1
            print(f"  ✓ 港股日线 {code}  {len(df)} 行  {df.index[0].date()}~{df.index[-1].date()}", flush=True)
        except Exception as e:
            print(f"  ✗ 港股日线 {code}  {str(e)[:50]}", flush=True)
        time.sleep(0.5)
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
        codes: 显式指定 5 位港股代码列表；传 None 则用内置热门榜
    """
    import yfinance as yf

    if codes is None:
        codes = [
            "00700", "09988", "01024", "01810", "09626", "09888", "09999",
            "03690", "01211", "02015", "02318", "02628", "01299", "00005",
            "00011", "00388", "00883", "00941", "00981", "01347",
        ]
    else:
        # 显式清单：补 0 到 5 位
        codes = [str(c).strip().zfill(5) for c in codes if str(c).strip()]

    label = f"港股{interval}"
    out = HK_5M_DIR if interval == "5m" else HK_1M_DIR
    ensure_data_dirs()
    ok = 0
    failed = []
    # yfinance 对 Yahoo 限流敏感，间隔要够大 + 限流退避重试
    base_sleep = 2.0  # 基础间隔 2s（比旧 0.3s 慢但稳）
    for code in codes[:top_n]:
        path = out / f"{code}.csv"
        if path.exists() and path.stat().st_size > 1000:
            ok += 1
            continue
        yf_code = _fmt_yf_hk(code)
        done = False
        for attempt in range(4):  # 最多 4 次重试
            try:
                df = yf.download(yf_code, period=period, interval=interval, progress=False)
                if df.empty:
                    # 退市/无数据，不重试直接跳
                    break
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df[["Open", "High", "Low", "Close", "Volume"]].rename(
                    columns={"Open": "open", "High": "high", "Low": "low",
                             "Close": "close", "Volume": "volume"}
                )
                df.to_csv(path)
                ok += 1
                print(f"  ✓ {label} {code}  {len(df)} 行", flush=True)
                done = True
                break
            except Exception as e:
                msg = str(e)[:80]
                is_rate_limit = "RateLimit" in msg or "Too Many" in msg or "429" in msg
                if is_rate_limit and attempt < 3:
                    wait = 5 * (attempt + 1)  # 5s, 10s, 15s 退避
                    print(f"  ⏳ {label} {code} 限流，等 {wait}s 后重试（第 {attempt+1}/3 次）", flush=True)
                    time.sleep(wait)
                    continue
                # 非限流或重试耗尽
                if not is_rate_limit:
                    break  # 退市/无数据不重试
                # 限流重试耗尽
                if attempt == 3:
                    failed.append(code)
                    print(f"  ✗ {label} {code} 限流 4 次仍失败，跳过", flush=True)
                else:
                    failed.append(code)
                    print(f"  ✗ {label} {code}  {msg}", flush=True)
                break
        if not done:
            continue
        time.sleep(base_sleep)
    print(f"  {label}完成: {ok}/{min(top_n, len(codes))}  失败 {len(failed)} 只")
    if failed:
        print(f"  失败清单: {','.join(failed[:20])}{'...' if len(failed) > 20 else ''}")


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

    out = US_DAILY_DIR
    ensure_data_dirs()
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