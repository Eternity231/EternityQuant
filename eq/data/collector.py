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
import requests


# ---------- 东财 push2his 统一 K 线拉取器（A/港/美三市场统一接口，免费无 key） ----------
#
# 真实用法（浏览器验证可用，沙盒代理可能屏）:
#   URL: https://push2his.eastmoney.com/api/qt/stock/kline/get
#   secid 映射: A 股沪=1.<code>, 深=0.<code>; 港股=116.<5位code>; 美股 NASDAQ=105.<sym>, NYSE=106.<sym>
#   klt: 101 日 K / 102 周 / 103 月 / 1 1分钟 / 5 5分钟 / 15 / 30 / 60
#   fqt: 0 不复权 / 1 前复权 / 2 后复权
#   beg/end: YYYYMMDD
#   fields2=f51..f61 = 日期,开,收,高,低,量,额,振幅,涨跌幅,涨跌额,换手率
#   JSONP 剥离: cb="" 时无包裹直接 JSON；否则剥 cb(...) 前后
#
# 三市场统一接入，替代分散的 akshare(港) / yfinance(美) 多套源
_EM_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
_EM_UT = "fa5fd1943c7b386f172d6893dbfba10b"  # 东财固定 ut token（公开）
_EM_FIELDS2 = "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"  # 11 列 K 线
_EM_KLT = {"daily": "101", "weekly": "102", "monthly": "103",
            "1": "1", "5": "5", "15": "15", "30": "30", "60": "60"}
_EM_COLS = ["date", "open", "close", "high", "low", "volume", "amount",
             "amplitude", "pct_change", "change", "turnover"]


def _em_secid(code: str, market: str) -> str:
    """把项目内标准 code 映射成东财 secid。

    Args:
        code: 项目内标准 code
            A 股: "SH600519" / "SZ000001" / "600519.SH"
            港股: "00700" / "09988"（5 位数字，前导零保留）
            美股: "AAPL" / "MSFT"
        market: "a" / "hk" / "us"
    """
    c = code.upper().replace(".SH", "").replace(".SZ", "").replace(".US", "").replace(".HK", "")
    if market == "a":
        # 沪=1. 深=0.（含北交所 SZ）
        if c.startswith("SH"):
            return "1." + c[2:]
        if c.startswith("SZ"):
            return "0." + c[2:]
        if c.startswith("6"):  # 裸代码默认沪
            return "1." + c
        return "0." + c
    if market == "hk":
        # 港股固定 116.<5位数字>，保留前导零
        return "116." + c.zfill(5)
    if market == "us":
        # NASDAQ=105, NYSE=106；无 key 时用 105 兜底（美股代码大概率 NASDAQ）
        return "105." + c
    raise ValueError(f"未知 market: {market}")


def _em_kline(code: str, market: str, start: str, end: str,
              period: str = "daily", adjust: str = "qfq") -> pd.DataFrame:
    """东财统一日 K/分钟 K 拉取器，返回 OHLCV DataFrame。

    Args:
        code: 项目内标准 code（同 _em_secid）
        market: "a" / "hk" / "us"
        start/end: "YYYY-MM-DD" 或 "YYYYMMDD"（统一转 YYYYMMDD）
        period: "daily" / "weekly" / "monthly" / "1" / "5" / "15" / "30" / "60"
        adjust: "qfq" 前复权（默认）/ "hfq" 后复权 / "" 不复权
    Returns:
        DataFrame[date, open, close, high, low, volume, amount, pct_change, turnover]
        失败时返回空 DataFrame（调用方 fallback 到 akshare/yfinance）
    """
    beg = start.replace("-", "")
    end_ = end.replace("-", "")
    klt = _EM_KLT.get(period, "101")
    fqt = {"qfq": "1", "hfq": "2"}.get(adjust, "0")
    params = {
        "cb": "",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": _EM_FIELDS2,
        "ut": _EM_UT,
        "klt": klt,
        "fqt": fqt,
        "secid": _em_secid(code, market),
        "beg": beg,
        "end": end_,
        "lmt": "8000",  # 单次上限，日 K 足盖 30 年
        "_": str(int(time.time() * 1000)),
    }
    try:
        r = requests.get(_EM_URL, params=params, timeout=20,
                         headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
        r.encoding = "utf-8"
        txt = r.text.strip()
        # JSONP 剥离（cb="" 时一般无包裹，但东财偶尔回 cb("")）
        if txt.startswith("(") and txt.endswith(");"):
            txt = txt[1:-2]
        if txt.startswith('"') or txt.startswith("'"):
            txt = txt.strip('"').strip("'")
        j = pd.io.json.loads(txt) if hasattr(pd.io.json, "loads") else __import__("json").loads(txt)
        kl = j.get("data", {}).get("klines", [])
        if not kl:
            return pd.DataFrame()
        rows = [row.split(",") for row in kl]
        df = pd.DataFrame(rows, columns=_EM_COLS)
        # 数值列转 float
        for col in ["open", "close", "high", "low", "amount", "amplitude", "pct_change", "change", "turnover"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")
        df["date"] = pd.to_datetime(df["date"])  # 日 K 是 YYYY-MM-DD，分钟 K 是 YYYY-MM-DD HH:MM
        df = df.sort_values("date").reset_index(drop=True)
        # 裁剪到 [start, end]（东财 beg/end 含端点，再核一次防越界）
        if beg:
            df = df[df["date"] >= pd.Timestamp(beg)]
        if end_:
            df = df[df["date"] <= pd.Timestamp(end_)]
        return df[["date", "open", "close", "high", "low", "volume", "amount", "pct_change", "turnover"]]
    except Exception:
        return pd.DataFrame()  # 调用方 fallback



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
            # 主源: 东财 push2his 统一接口（免费无 key，A/港/美同源）
            df = _em_kline(code, "hk", start, end, period="daily", adjust="qfq")
            if df.empty:
                # fallback: akshare 新浪源（东财被限流时）
                import akshare as ak
                df = ak.stock_hk_daily(symbol=code, adjust="qfq")
                if df.empty:
                    continue
                df = df.rename(columns={"date": "date"}).set_index("date")
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                df = df[["open", "high", "low", "close", "volume"]].reset_index()
                if start:
                    df = df[df["date"] >= pd.Timestamp(start)]
                if end:
                    df = df[df["date"] <= pd.Timestamp(end)]
            else:
                df = df.set_index("date").rename(columns={"close": "close"})[["open", "close", "high", "low", "volume"]]
            if df.empty:
                continue
            df.to_csv(path)
            ok += 1
            print(f"  ✓ 港股日线 {code}  {len(df)} 行  {df.index[0].date()}~{df.index[-1].date()}", flush=True)
        except Exception as e:
            print(f"  ✗ 港股日线 {code}  {str(e)[:50]}", flush=True)
        time.sleep(0.3)  # 东财源比 akshare 快，间隔从 0.5→0.3
    print(f"  港股日线完成: {ok}/{min(top_n, len(codes))}")


def collect_hk_minute(
    codes: list[str] | None = None,
    top_n: int = 200,
    interval: str = "5m",
    period: str = "2mo",
):
    """港股分钟线（yfinance，8s 间隔防限流）。

    Args:
        codes: 显式指定 5 位港股代码列表；传 None 则用内置热门榜
    """
    return _collect_hk_minute_yf(codes=codes, top_n=top_n, interval=interval, period=period)


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
            # 主源: 东财 push2his 统一接口（免费无 key，A/港/美同源）
            df = _em_kline(code, "us", start, end, period="daily", adjust="qfq")
            if df.empty:
                # fallback: yfinance（东财被限流时）
                import yfinance as yf
                df = yf.download(code, start=start, end=end, progress=False)
                if df.empty:
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df[["Open", "High", "Low", "Close", "Volume"]].rename(
                    columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
                ).reset_index().rename(columns={"Date": "date"})
            else:
                df = df.set_index("date")[["open", "close", "high", "low", "volume"]]
            if df.empty:
                continue
            df.to_csv(path)
            ok += 1
            print(f"  ✓ 美股日线 {code}  {len(df)} 行", flush=True)
        except Exception as e:
            print(f"  ✗ 美股日线 {code}  {str(e)[:50]}", flush=True)
        time.sleep(0.3)  # 东财源比 yfinance 快，间隔从 0.3→0.3
    print(f"  美股日线完成: {ok}/{min(top_n, len(codes))}")


def collect_a_share(start: str = "2026-01-01", universe: str = "csi300", workers: int = 10, extra_codes: list[str] | None = None):
    """A 股日线（腾讯 API → qlib .bin）。

    ``extra_codes`` 用于「单只股票 + 预设指数共同下载并训练」：
    传 ``["SH600519"]`` 等额外代码，它们会与 ``universe``（如 csi500）
    的成分股合并去重后一起下载、写进同一份 instruments 文件、一起训练。
    """
    from eq.strategy.factors.ml_data_updater import update_qlib_data
    result = update_qlib_data(
        start=start, universe=universe, workers=workers,
        extra_codes=extra_codes, verbose=True,
    )
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