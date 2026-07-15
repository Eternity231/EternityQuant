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
    """港股分钟线（东财 API + curl_cffi 浏览器指纹伪装，免翻墙免限流）。

    Args:
        interval: 1m | 5m | 15m | 30m | 60m
        period: 仅兼容旧调用，实际由东财返回全部历史数据
        codes: 显式指定 5 位港股代码列表；传 None 则用内置热门榜
    """
    try:
        from curl_cffi import requests as curl_requests
    except ImportError:
        print("  ⚠ curl_cffi 未安装，回退到 yfinance（pip install curl_cffi）", flush=True)
        return _collect_hk_minute_yf(codes=codes, top_n=top_n, interval=interval, period=period)

    if codes is None:
        codes = [
            "00700", "09988", "01024", "01810", "09626", "09888", "09999",
            "03690", "01211", "02015", "02318", "02628", "01299", "00005",
            "00011", "00388", "00883", "00941", "00981", "01347",
        ]
    else:
        codes = [str(c).strip().zfill(5) for c in codes if str(c).strip()]

    period_map = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "60m": "60"}
    klt = period_map.get(interval, "5")

    label = f"港股{interval}"
    out = HK_5M_DIR if interval == "5m" else HK_1M_DIR
    ensure_data_dirs()
    ok = 0
    failed = []

    # 东财不限流，0.5s 间隔足够
    for code in codes[:top_n]:
        path = out / f"{code}.csv"
        if path.exists() and path.stat().st_size > 1000:
            ok += 1
            continue

        url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        params = {
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            "klt": klt,
            "fqt": "0",
            "secid": f"116.{code}",
            "beg": "0",
            "end": "20500000",
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
            "Referer": f"https://quote.eastmoney.com/hk/{code}.html",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }

        try:
            r = curl_requests.get(
                url, params=params, headers=headers,
                impersonate="chrome146", timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            klines = data.get("data", {}).get("klines", [])
            if not klines:
                print(f"  ✗ {label} {code}  东财返回空数据", flush=True)
                failed.append(code)
                continue

            # 解析东财格式：时间,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率
            rows = []
            for item in klines:
                cols = item.split(",")
                if len(cols) >= 6:
                    rows.append({
                        "Date": cols[0],
                        "open": float(cols[1]),
                        "high": float(cols[3]),
                        "low": float(cols[4]),
                        "close": float(cols[2]),
                        "volume": float(cols[5]),
                    })
            if not rows:
                print(f"  ✗ {label} {code}  解析后无有效行", flush=True)
                failed.append(code)
                continue

            df = pd.DataFrame(rows)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()
            df.to_csv(path)
            ok += 1
            print(f"  ✓ {label} {code}  {len(df)} 行  {df.index[0]}~{df.index[-1]}", flush=True)

        except Exception as e:
            print(f"  ✗ {label} {code}  {type(e).__name__}: {str(e)[:80]}", flush=True)
            failed.append(code)

        time.sleep(0.5)

    print(f"  {label}完成: {ok}/{min(top_n, len(codes))}  失败 {len(failed)} 只")
    if failed:
        print(f"  失败清单: {','.join(failed[:20])}{'...' if len(failed) > 20 else ''}")


def _collect_hk_minute_yf(
    codes: list[str] | None = None,
    top_n: int = 200,
    interval: str = "5m",
    period: str = "2mo",
):
    """港股分钟线 fallback：yfinance（Yahoo 源，限流严重）。"""
    import yfinance as yf

    if codes is None:
        codes = [
            "00700", "09988", "01024", "01810", "09626", "09888", "09999",
            "03690", "01211", "02015", "02318", "02628", "01299", "00005",
            "00011", "00388", "00883", "00941", "00981", "01347",
        ]
    else:
        codes = [str(c).strip().zfill(5) for c in codes if str(c).strip()]

    label = f"港股{interval}"
    out = HK_5M_DIR if interval == "5m" else HK_1M_DIR
    ensure_data_dirs()
    ok = 0
    failed = []
    base_sleep = 6.0
    consecutive_ok = 0
    for code in codes[:top_n]:
        path = out / f"{code}.csv"
        if path.exists() and path.stat().st_size > 1000:
            ok += 1
            continue
        yf_code = _fmt_yf_hk(code)
        done = False
        for attempt in range(4):
            try:
                df = yf.download(yf_code, period=period, interval=interval, progress=False)
                if df.empty:
                    break
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df[["Open", "High", "Low", "Close", "Volume"]].rename(
                    columns={"Open": "open", "High": "high", "Low": "low",
                             "Close": "close", "Volume": "volume"}
                )
                df.to_csv(path)
                ok += 1
                consecutive_ok += 1
                print(f"  ✓ {label} {code}  {len(df)} 行", flush=True)
                done = True
                break
            except Exception as e:
                msg = str(e)[:100]
                is_rate_limit = "RateLimit" in msg or "Too Many" in msg or "429" in msg
                if is_rate_limit and attempt < 3:
                    wait = 30 * (attempt + 1)
                    print(f"  ⏳ {label} {code} 限流，等 {wait}s 后重试（第 {attempt+1}/3 次）", flush=True)
                    time.sleep(wait)
                    consecutive_ok = 0
                    continue
                if not is_rate_limit:
                    break
                failed.append(code)
                print(f"  ✗ {label} {code} 限流 4 次仍失败，跳过", flush=True)
                break
        if not done:
            continue
        dynamic_sleep = max(4.0, base_sleep - (consecutive_ok // 5) * 1.0)
        time.sleep(dynamic_sleep)
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