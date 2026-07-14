"""qlib A 股数据更新器：baostock 拉日线 → 转 qlib .bin 格式续期。

qlib 本地数据集截至 2020-09-25，本更新器把数据续到最新（今天）。
baostock 是 TCP 长连接（多线程不安全），用 multiprocessing.Pool 真并行，
每个子进程独立 login/logout。

用法：
    eq ml update-data --start 2020-09-28 --universe csi300 --workers 10
"""

from __future__ import annotations

import datetime as dt
import struct
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_QLIB_DATA_DIR = (Path(__file__).resolve().parent.parent.parent.parent / ".qlib_data" / "cn_data")
_FEATURES = ["open", "high", "low", "close", "volume", "factor", "change"]
_FLOAT32_NAN = np.float32(np.nan)


# ---------- 模块级工作函数（multiprocessing.Pool 在 Windows spawn 模式下必须可 pickle） ----------

def _worker_init():
    """每个子进程启动时 baostock login。"""
    import baostock as _bs
    _bs.login()


def _worker_finish():
    """每个子进程完成后 baostock logout。"""
    import baostock as _bs
    _bs.logout()


def _proc_one_stock(code: str, start: str, end: str, new_days: tuple, qlib_feats_dir: str, expected_floats: int = 0) -> bool:
    """子进程内处理一只票。失败自动重试最多 3 次（baostock 限流常见）。

    返回 True=成功 False=失败（停牌或异常）。
    """
    import baostock as _bs
    from pathlib import Path
    import pandas as pd
    import time as _time

    inst_dir = Path(qlib_feats_dir) / code.lower()
    inst_dir.mkdir(parents=True, exist_ok=True)

    # 断点续传检查：若已完整下载则跳过，不完整则清空重写（崩溃恢复）
    days_list = list(new_days)
    close_bin = inst_dir / "close.day.bin"
    if close_bin.exists() and expected_floats > 0:
        actual = close_bin.stat().st_size // 4
        if actual >= expected_floats:
            return True
        # 不完整（崩溃/断点）：清空重写
        for feat in _FEATURES:
            bp = inst_dir / f"{feat}.day.bin"
            if bp.exists():
                bp.unlink()

    # qlib SH600000 → baostock sh.600000
    if code.startswith("SH"):
        bs_code = f"sh.{code[2:]}"
    elif code.startswith("SZ"):
        bs_code = f"sz.{code[2:]}"
    else:
        return False

    for attempt in range(5):  # 指数退避
        try:
            if attempt > 0:
                _time.sleep(2 ** attempt)  # 退避：2/4/8/16/32s
            rs = _bs.query_history_k_data_plus(
                bs_code, "date,open,high,low,close,volume,preclose,adjustflag,turn",
                start_date=start, end_date=end, frequency="d", adjustflag="2",
            )
            if rs.error_code != "0":
                if attempt >= 4:
                    return False
            rows_list = []
            while rs.next():
                rows_list.append(rs.get_row_data())
            if not rows_list:
                return False
            df = pd.DataFrame(rows_list, columns=["date", "open", "high", "low", "close", "volume", "preclose", "adjustflag", "turn"])
            for col in ["open", "high", "low", "close", "volume", "preclose"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["factor"] = 1.0
            df["change"] = (df["close"] - df["preclose"]) / df["preclose"]
            df = df.set_index("date")[["open", "high", "low", "close", "volume", "factor", "change"]]
            # 对齐日历
            days_list = list(new_days)
            df = df.reindex(days_list)
            # 写 .bin
            for feat in _FEATURES:
                bin_path = inst_dir / f"{feat}.day.bin"
                vals = df[feat].tolist() if feat in df.columns else [float("nan")] * len(days_list)
                vals = [float("nan") if v != v or v is None else v for v in vals]
                _append_bin(bin_path, vals)
            return True
        except Exception:
            if attempt >= 2:
                return False
            continue
    return False


def _proc_batch(args: tuple) -> tuple:
    """子进程内处理一批代码，返回(成功数, 失败数)。
    
    args: (codes, start, end, new_days, feats_dir_str)
    """
    codes, start, end, new_days, feats_dir_str, expected_floats = args
    _worker_init()
    ok = fail = 0
    for code in codes:
        if _proc_one_stock(code, start, end, new_days, feats_dir_str, expected_floats=expected_floats):
            ok += 1
        else:
            fail += 1
    _worker_finish()
    return ok, fail


def _append_bin(bin_path: Path, new_values: list[float]) -> None:
    """给 .bin 文件追加 float32 数据。停牌日写 NaN。"""
    arr = np.array(new_values, dtype=np.float32)
    with open(bin_path, "ab") as f:
        f.write(arr.tobytes())


# ---------- 内部工具函数 ----------

def _bs_instruments(universe: str = "csi300") -> list[str]:
    """baostock 拉成分股列表，返回 qlib 格式代码列表（SH600000/sz000001）。"""
    import baostock as bs
    if universe == "csi300":
        rs = bs.query_hs300_stocks()
    elif universe == "csi500":
        rs = bs.query_zz500_stocks()
    elif universe == "sz50":
        rs = bs.query_sz50_stocks()
    else:
        # all：沪深京全 A，用 akshare 新浪源
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
    # row: [date, code, name] → code 如 "sh.600000" → "SH600000"
    return [r[1].replace(".", "").upper() for r in rows] if rows else []


def _trading_days(start: str, end: str) -> list[str]:
    """拉 start~end 的交易日列表（baostock query_trade_dates）。"""
    import baostock as bs
    rs = bs.query_trade_dates(start_date=start, end_date=end)
    if rs.error_code != "0":
        raise RuntimeError(f"baostock 拉交易日失败：{rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    return [r[0] for r in rows if r[1] == "1"]


# ---------- 主入口 ----------

def update_qlib_data(
    start: str = "2020-09-28",
    end: str | None = None,
    universe: str = "csi300",
    instruments: list[str] | None = None,
    workers: int = 10,
    verbose: bool = True,
) -> dict:
    """把 qlib 本地数据从 start 续到 end（默认今天）。多进程并行拉。

    Args:
        start: 续期起始日（YYYY-MM-DD），默认 2020-09-28（接 2020-09-25）
        end: 续期结束日，默认今天
        universe: csi300/csi500/all，成分股列表
        instruments: 显式指定代码列表，覆盖 universe
        workers: 并行进程数（baostock TCP 长连接，多进程真并行，默认 10）
        verbose: 打进度
    Returns:
        {"days_added": int, "instruments_updated": int, "features_per_inst": int}
    """
    if end is None:
        end = dt.date.today().isoformat()

    import baostock as bs
    bs.login()
    try:
        new_days = _trading_days(start, end)
        if not new_days:
            return {"days_added": 0, "instruments_updated": 0, "features_per_inst": 0}
        if verbose:
            print(f"续期 {start} ~ {end}：{len(new_days)} 个交易日", flush=True)

        # 续日历文件
        cal_path = _QLIB_DATA_DIR / "calendars" / "day.txt"
        existing_cal = cal_path.read_text().strip().split("\n") if cal_path.exists() else []
        new_cal_days = [d for d in new_days if d not in existing_cal]
        with open(cal_path, "a") as f:
            for d in new_cal_days:
                f.write(d + "\n")
        if verbose:
            print(f"日历续 {len(new_cal_days)} 日（去重后）", flush=True)

        # 拉成分股
        if instruments is None:
            instruments = _bs_instruments(universe)
        if verbose:
            print(f"成分股 {len(instruments)} 只（universe={universe}），{workers} 进程并行", flush=True)

        # 多进程并行拉
        feats_dir = _QLIB_DATA_DIR / "features"
        feats_dir.mkdir(parents=True, exist_ok=True)
        total = len(instruments)
        import time as _time
        import multiprocessing as _mp

        # 分批：每批 workers 个代码，每个子进程跑一批
        batch_size = max(1, workers)
        batches = [instruments[i:i + batch_size] for i in range(0, total, batch_size)]
        # 预期 float32 个数 = 原始日历行数 + new_days（.bin 文件应含全部日历日的 float）
        expected_floats_val = len(existing_cal) + len(new_days)
        batch_args = [(b, start, end, tuple(new_days), str(feats_dir), expected_floats_val) for b in batches]

        _t0 = _time.time()
        done = _mp.Value("i", 0)
        ok = _mp.Value("i", 0)
        fail = _mp.Value("i", 0)
        done_count = [0]  # 已返回批次计数

        with _mp.Pool(processes=workers) as pool:
            results = pool.imap_unordered(_proc_batch, batch_args)
            for ok_cnt, fail_cnt in results:
                done_count[0] += 1
                with ok.get_lock():
                    ok.value += ok_cnt
                    fail.value += fail_cnt
                    done.value = ok.value + fail.value
                if verbose:
                    _elapsed = _time.time() - _t0
                    _pct = min(done.value / total * 100, 100.0)
                    _speed = done.value / _elapsed if _elapsed > 0 else 0
                    _eta = (total - done.value) / _speed if _speed > 0 else 0
                    print(f"\r  进度 {_pct:5.1f}%  ({done.value}/{total})  ✓{ok.value} ✗{fail.value}  已用 {_elapsed:.0f}s  ETA {_eta:.0f}s", end="", flush=True)

        if verbose:
            _total_elapsed = _time.time() - _t0
            print(f"\n完成：{ok.value} 只票 ✓，{fail.value} 只失败 ✗  × {len(_FEATURES)} 特征 × {len(new_days)} 日  ({_total_elapsed:.0f}s)", flush=True)
    finally:
        bs.logout()

    return {
        "days_added": len(new_cal_days),
        "instruments_updated": ok.value,
        "features_per_inst": len(_FEATURES),
        "trading_days": len(new_days),
    }
