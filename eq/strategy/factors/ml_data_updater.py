"""qlib A 股数据更新器：baostock 拉日线 → 转 qlib .bin 格式续期。

qlib 本地数据集截至 2020-09-25，本更新器把数据续到最新（今天）：
1. 日历续期：2020-09-26 ~ 今天 的交易日写入 calendars/day.txt
2. 特征续期：每只票的 open/high/low/close/volume/factor/change 转 .day.bin 续期
3. 成分股续期：csi300/csi500/all 的 instruments 列表更新

.bin 格式：float32，无 header，按日历顺序。停牌日该位置写 NaN（float32 的 nan）。
baostock 生命周期：login() → query → logout()，避免资源泄漏。

用法：
    from eq.strategy.factors.ml_data_updater import update_qlib_data
    update_qlib_data(start="2020-09-28", end="2026-07-13", universe="csi300")
"""

from __future__ import annotations

import datetime as dt
import struct
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_QLIB_DATA_DIR = Path.home() / ".qlib" / "qlib_data" / "cn_data"
_FEATURES = ["open", "high", "low", "close", "volume", "factor", "change"]
_FLOAT32_NAN = np.float32(np.nan)


def _bs_instruments(universe: str = "csi300") -> list[str]:
    """baostock 拉成分股列表，返回 qlib 格式代码列表（SH600519/sz000001）。"""
    import baostock as bs
    if universe == "csi300":
        rs = bs.query_hs300_stocks()  # 沪深300
    elif universe == "csi500":
        rs = bs.query_zz500_stocks()
    elif universe == "sz50":
        rs = bs.query_sz50_stocks()
    else:
        # all：沪深京全 A，用 akshare 新浪源拉全市场代码
        import akshare as ak
        df = ak.stock_zh_a_spot()
        codes = []
        for raw in df["代码"].tolist():
            prefix = raw[:2].lower()
            suffix = {"sh": "SH", "sz": "SZ", "bj": "BJ"}.get(prefix)
            if suffix:
                codes.append(f"{suffix}{raw[2:]}")
        return codes
    if rs.error_code != "0":
        raise RuntimeError(f"baostock 拉成分股失败：{rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    # row_data[1] 是代码如 "sh.600000"，转 "SH600000"
    return [r[1].replace(".", "").upper() for r in rows] if rows else []


def _bs_query_k_data(code: str, start: str, end: str) -> pd.DataFrame:
    """baostock 拉单只票日线，返回 DataFrame(open/high/low/close/volume/factor/change)。

    baostock 代码格式：sh.600000 / sz.000001（qlib 是 SH600000/sz000001，转一下）。
    """
    import baostock as bs
    # qlib SH600000 → baostock sh.600000
    if code.startswith("SH"):
        bs_code = f"sh.{code[2:]}"
    elif code.startswith("SZ"):
        bs_code = f"sz.{code[2:]}"
    else:
        return pd.DataFrame()
    rs = bs.query_history_k_data_plus(
        bs_code,
        "date,open,high,low,close,volume,preclose,adjustflag,turn",
        start_date=start, end_date=end, frequency="d", adjustflag="2",  # 前复权
    )
    if rs.error_code != "0":
        return pd.DataFrame()
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "preclose", "adjustflag", "turn"])
    # 转 float，空字符串 → NaN
    for col in ["open", "high", "low", "close", "volume", "preclose"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # factor：复权因子，baostock adjustflag=2 前复权，factor 简化为 1.0（已复权）
    df["factor"] = 1.0
    # change：涨跌幅 = (close - preclose) / preclose
    df["change"] = (df["close"] - df["preclose"]) / df["preclose"]
    return df.set_index("date")[["open", "high", "low", "close", "volume", "factor", "change"]]


def _trading_days(start: str, end: str) -> list[str]:
    """拉 start~end 的交易日列表（baostock query_trade_base）。"""
    import baostock as bs
    rs = bs.query_trade_dates(start_date=start, end_date=end)
    if rs.error_code != "0":
        raise RuntimeError(f"baostock 拉交易日失败：{rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    # row: [date, is_trading_day]，只取交易日
    return [r[0] for r in rows if r[1] == "1"]


def _append_bin(bin_path: Path, new_values: list[float]) -> None:
    """给 .bin 文件追加 float32 数据。停牌日写 NaN。"""
    arr = np.array(new_values, dtype=np.float32)
    with open(bin_path, "ab") as f:  # append binary
        f.write(arr.tobytes())


def _build_calendar_cache() -> list[str]:
    """读现有 calendars/day.txt 全量日历。"""
    p = _QLIB_DATA_DIR / "calendars" / "day.txt"
    if not p.exists():
        return []
    return p.read_text().strip().split("\n")


def update_qlib_data(
    start: str = "2020-09-28",
    end: str | None = None,
    universe: str = "csi300",
    instruments: list[str] | None = None,
    verbose: bool = True,
) -> dict:
    """把 qlib 本地数据从 start 续到 end（默认今天）。

    Args:
        start: 续期起始日（YYYY-MM-DD），默认 2020-09-28（接 2020-09-25）
        end: 续期结束日，默认今天
        universe: csi300/csi500/all，成分股列表
        instruments: 显式指定代码列表，覆盖 universe
        verbose: 打进度
    Returns:
        {"days_added": int, "instruments_updated": int, "features_per_inst": int}
    """
    if end is None:
        end = dt.date.today().isoformat()

    # 1. 拉交易日历（start~end）
    import baostock as bs
    bs.login()
    try:
        new_days = _trading_days(start, end)
        if not new_days:
            return {"days_added": 0, "instruments_updated": 0, "features_per_inst": 0}
        if verbose:
            print(f"续期 {start} ~ {end}：{len(new_days)} 个交易日", flush=True)

        # 2. 续日历文件
        cal_path = _QLIB_DATA_DIR / "calendars" / "day.txt"
        existing_cal = cal_path.read_text().strip().split("\n") if cal_path.exists() else []
        # 去重：新日历里可能和现有的末尾重叠
        new_cal_days = [d for d in new_days if d not in existing_cal]
        with open(cal_path, "a") as f:
            for d in new_cal_days:
                f.write(d + "\n")
        if verbose:
            print(f"日历续 {len(new_cal_days)} 日（去重后）", flush=True)

        # 3. 拉成分股列表
        if instruments is None:
            instruments = _bs_instruments(universe)
        if verbose:
            print(f"成分股 {len(instruments)} 只（universe={universe}）", flush=True)

        # 4. 每只票拉日线 → 转 .bin 续期
        feats_dir = _QLIB_DATA_DIR / "features"
        feats_dir.mkdir(parents=True, exist_ok=True)
        updated = 0
        total = len(instruments)
        import time as _time
        _t0 = _time.time()
        for i, code in enumerate(instruments):
            # qlib 代码 SH600519 → features/sh600519/
            inst_dir = feats_dir / code.lower()
            inst_dir.mkdir(parents=True, exist_ok=True)
            # 拉日线
            df = _bs_query_k_data(code, start, end)
            if df.empty:
                continue
            # 对齐日历：每个交易日一行，停牌写 NaN
            df = df.reindex(new_days)
            # 写每个特征的 .bin
            for feat in _FEATURES:
                bin_path = inst_dir / f"{feat}.day.bin"
                vals = df[feat].tolist() if feat in df.columns else [float("nan")] * len(new_days)
                # NaN 转 float32 nan
                vals = [float("nan") if v != v or v is None else v for v in vals]
                _append_bin(bin_path, vals)
            updated += 1
            if verbose:
                _elapsed = _time.time() - _t0
                _pct = (i + 1) / total * 100
                _speed = (i + 1) / _elapsed if _elapsed > 0 else 0
                _eta = (total - i - 1) / _speed if _speed > 0 else 0
                print(f"\r  进度 {_pct:5.1f}%  ({i+1}/{total})  已用 {_elapsed:.0f}s  ETA {_eta:.0f}s  当前 {code}", end="", flush=True)
        if verbose:
            _total_elapsed = _time.time() - _t0
            print(f"\n完成：{updated} 只票 × {len(_FEATURES)} 特征 × {len(new_days)} 日  ({_total_elapsed:.0f}s)", flush=True)
    finally:
        bs.logout()

    return {
        "days_added": len(new_cal_days),
        "instruments_updated": updated,
        "features_per_inst": len(_FEATURES),
        "trading_days": len(new_days),
    }
