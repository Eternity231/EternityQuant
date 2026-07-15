"""qlib A 股数据更新器：腾讯 API 拉日线 → 转 qlib .bin 格式续期。

替代被封的 baostock，改用腾讯财经 API（web.ifzq.gtimg.cn），
国内可直接访问，无需梯子、无需账号。

用法：
    eq ml update-data --start 2020-09-28 --universe csi300 --workers 10
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests as _req

from eq.data.paths import QLIB_CN_DATA_DIR as _QLIB_DATA_DIR, ensure_data_dirs

_FEATURES = ["open", "high", "low", "close", "volume", "factor", "change"]
_FLOAT32_NAN = np.float32(np.nan)

# 腾讯 API 基础 URL
_TENCENT_KLINE = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"


# ---------- 腾讯 API 数据获取 ----------

def _tencent_daily(code: str, start: str, end: str) -> pd.DataFrame | None:
    """从腾讯 API 拉单只股票日线，返回 DataFrame。

    Args:
        code: qlib 格式如 'SH600000' 或 'SZ000001'
        start: YYYY-MM-DD
        end: YYYY-MM-DD
    Returns:
        DataFrame with columns [open, high, low, close, volume], index=date
    """
    # 转腾讯格式：sh600000 / sz000001
    if code.startswith("SH"):
        tcode = "sh" + code[2:]
    elif code.startswith("SZ"):
        tcode = "sz" + code[2:]
    else:
        return None

    try:
        url = f"{_TENCENT_KLINE}?param={tcode},day,{start},{end},640,qfq"
        resp = _req.get(url, timeout=15)
        if resp.status_code != 200:
            print(f"  [DEBUG] {code} HTTP {resp.status_code}", flush=True)
            return None
        data = resp.json()
        if data.get("code") != 0:
            print(f"  [DEBUG] {code} 腾讯返回错误: code={data.get('code')}, msg={data.get('msg','')}", flush=True)
            return None
        # 腾讯返回 "qfqday"（前复权）或 "day"（不复权）
        stock_data = data.get("data", {}).get(tcode, {})
        bars = stock_data.get("qfqday") or stock_data.get("day")
        if not bars:
            print(f"  [DEBUG] {code} 腾讯返回空数据, keys={list(stock_data.keys()) if stock_data else 'no data'}", flush=True)
            return None
        # bars: [["date", "open", "close", "high", "low", "volume"], ...]
        rows = []
        for bar in bars:
            if len(bar) >= 6:
                rows.append({
                    "date": bar[0],
                    "open": float(bar[1]),
                    "close": float(bar[2]),
                    "high": float(bar[3]),
                    "low": float(bar[4]),
                    "volume": float(bar[5]),
                })
        if not rows:
            return None
        df = pd.DataFrame(rows).set_index("date")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df
    except Exception as e:
        print(f"  [DEBUG] {code} 异常: {e}", flush=True)
        return None


def _tencent_instruments(universe: str = "csi300") -> list[str]:
    """获取成分股列表。

    优先级：
    1. 本地缓存 ``{universe}_codes.txt``（逗号分隔的 qlib 代码如 ``SH600000``）
    2. akshare ``index_stock_cons_csindex``（中证指数公司官方源，最稳定）
    3. akshare ``index_stock_cons``（新浪源，老版 akshare 可用）
    4. 全 A 股：``stock_zh_a_spot``
    5. fallback：从已有 features/ 目录扫描

    Args:
        universe: csi300 | csi500 | sz50 | all
    Returns:
        qlib 格式代码列表，如 ``["SH600000", "SZ000001"]``
    """
    code_file = _QLIB_DATA_DIR / f"{universe}_codes.txt"
    if code_file.exists():
        codes = [c.strip() for c in code_file.read_text().strip().split(",") if c.strip()]
        if codes:
            return codes

    # akshare 指数代码映射
    name_map = {"csi300": "000300", "csi500": "000905", "sz50": "000016"}
    codes: list[str] = []

    # 2. 中证指数公司源（最稳定）
    if universe in name_map:
        try:
            import akshare as ak
            df = ak.index_stock_cons_csindex(symbol=name_map[universe])
            if df is not None and not df.empty:
                # 列名可能是「成分券代码」或「代码」
                code_col = "成分券代码" if "成分券代码" in df.columns else df.columns[0]
                for c in df[code_col].astype(str).str.zfill(6).tolist():
                    if c.startswith("6") or c.startswith("9"):
                        codes.append(f"SH{c}")
                    elif c.startswith(("0", "3")):
                        codes.append(f"SZ{c}")
                    elif c.startswith(("4", "8")):
                        codes.append(f"BJ{c}")
        except Exception as e:
            print(f"  [warn] index_stock_cons_csindex 失败: {e}", flush=True)

    # 3. 新浪源 fallback
    if not codes and universe in name_map:
        try:
            import akshare as ak
            name_zh = {"csi300": "沪深300", "csi500": "中证500", "sz50": "上证50"}[universe]
            df = ak.index_stock_cons(symbol=name_zh)
            if df is not None and not df.empty:
                for c in df.iloc[:, 0].astype(str).str.zfill(6).tolist():
                    if c.startswith("6") or c.startswith("9"):
                        codes.append(f"SH{c}")
                    elif c.startswith(("0", "3")):
                        codes.append(f"SZ{c}")
                    elif c.startswith(("4", "8")):
                        codes.append(f"BJ{c}")
        except Exception as e:
            print(f"  [warn] index_stock_cons 新浪源失败: {e}", flush=True)

    # 4. 全 A 股
    if universe == "all":
        try:
            import akshare as ak
            df = ak.stock_zh_a_spot()
            if df is not None and not df.empty:
                for c in df["代码"].astype(str).str.zfill(6).tolist():
                    if c.startswith(("4", "8", "9")):
                        codes.append(f"BJ{c}")
                    elif c.startswith("6"):
                        codes.append(f"SH{c}")
                    elif c.startswith(("0", "3")):
                        codes.append(f"SZ{c}")
        except Exception as e:
            print(f"  [warn] stock_zh_a_spot 失败: {e}", flush=True)

    # 5. 已有 features/ 目录扫描
    if not codes:
        feats_dir = _QLIB_DATA_DIR / "features"
        if feats_dir.exists():
            codes = [d.name.upper() for d in feats_dir.iterdir()
                     if d.is_dir() and d.name.startswith(("sh", "sz", "bj"))]

    if not codes:
        raise RuntimeError(
            f"无法获取 {universe} 成分股列表。\n"
            f"请先手动创建 {code_file}，内容为逗号分隔的股票代码，如：\n"
            f"SH600000,SZ000001,SH600004,..."
        )

    # 缓存到本地
    code_file.parent.mkdir(parents=True, exist_ok=True)
    code_file.write_text(",".join(codes))
    return codes


def _tencent_trading_days(start: str, end: str) -> list[str]:
    """从腾讯 API 获取交易日列表（通过查询沪深300指数）。"""
    try:
        url = f"{_TENCENT_KLINE}?param=sh000300,day,{start},{end},640,qfq"
        resp = _req.get(url, timeout=15)
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"腾讯 API 返回错误: {data}")
        stock_data = data.get("data", {}).get("sh000300", {})
        bars = stock_data.get("qfqday") or stock_data.get("day")
        if not bars:
            raise RuntimeError("腾讯 API 返回空数据")
        days = [bar[0] for bar in bars if len(bar) >= 1]
        return sorted(set(days))
    except Exception as e:
        raise RuntimeError(f"腾讯 API 拉交易日失败：{e}")


# ---------- 股池列表（新浪源） ----------

def _bs_instruments(universe: str = "csi300") -> list[str]:
    """获取成分股列表（兼容旧接口名，实际走腾讯/新浪）。"""
    return _tencent_instruments(universe)


def _trading_days(start: str, end: str) -> list[str]:
    """获取交易日列表（兼容旧接口名，实际走腾讯 API）。"""
    return _tencent_trading_days(start, end)


# ---------- 单只股票处理（用于 multiprocessing） ----------

def _proc_one_stock(code, start, end, new_days, qlib_feats_dir, expected_floats=0):
    """用腾讯 API 下载单只股票日线，转 qlib .bin 格式。

    Args:
        code: 如 "SH600000"
        start, end: 日期范围
        new_days: 交易日列表
        qlib_feats_dir: 特征输出目录
        expected_floats: 每个 .bin 文件预期的 float32 数量
    Returns:
        bool: 是否成功
    """
    import time as _t
    from pathlib import Path

    inst_dir = Path(qlib_feats_dir) / code.lower()
    inst_dir.mkdir(parents=True, exist_ok=True)
    days_list = list(new_days)

    # 断点续传检查
    close_bin = inst_dir / "close.day.bin"
    if close_bin.exists() and expected_floats > 0:
        actual = close_bin.stat().st_size // 4
        if actual >= expected_floats:
            return True
        for feat in _FEATURES:
            bp = inst_dir / f"{feat}.day.bin"
            if bp.exists():
                bp.unlink()

    for attempt in range(3):
        if attempt > 0:
            _t.sleep(2 ** attempt)  # 指数退避 2s, 4s
        try:
            # 请求间隔，避免被限流
            if attempt == 0:
                _t.sleep(0.1)  # 每只股票间隔 100ms
            df = _tencent_daily(code, start, end)
            if df is None or df.empty:
                if attempt >= 2:
                    print(f"  [DEBUG] {code} 第{attempt+1}次尝试无数据", flush=True)
                return False
            # 按交易日对齐
            df = df.reindex(pd.to_datetime(days_list))
            # 前向填充 NaN（停牌日沿用上一交易日数据）
            df = df.ffill()
            # 如果开头还有 NaN，用后续第一个有效值填充
            df = df.bfill()
            # 如果全部是 NaN，放弃这只股票
            if df["close"].isna().all():
                return False
            df["factor"] = 1.0
            df["change"] = df["close"].pct_change(fill_method=None).fillna(0.0)
            # 转 .bin
            for feat in _FEATURES:
                vals = df[feat].tolist() if feat in df.columns else [float("nan")] * len(days_list)
                vals = [float("nan") if v != v or v is None else v for v in vals]
                _append_bin(inst_dir / f"{feat}.day.bin", vals)
            return True
        except Exception as e:
            if attempt >= 2:
                print(f"  [DEBUG] {code} 第{attempt+1}次异常: {e}", flush=True)
                return False
    return False


def _proc_batch(args):
    """批量处理一批股票（腾讯 API 无连接池概念，直接串行请求）。"""
    codes, start, end, new_days, qlib_feats_dir, expected_floats = args
    ok = fail = 0
    for code in codes:
        try:
            if _proc_one_stock(code, start, end, new_days, qlib_feats_dir, expected_floats):
                ok += 1
            else:
                fail += 1
        except Exception:
            fail += 1
    return ok, fail


def _append_bin(bin_path: Path, new_values: list[float]) -> None:
    """给 .bin 文件追加 float32 数据。停牌日写 NaN。"""
    arr = np.array(new_values, dtype=np.float32)
    with open(bin_path, "ab") as f:
        f.write(arr.tobytes())


# ---------- 主入口 ----------

def update_qlib_data(
    start: str = "2020-09-28",
    end: str | None = None,
    universe: str = "csi300",
    instruments: list[str] | None = None,
    workers: int = 5,  # 默认 5 进程，避免 API 限流和 CPU 爆满
    verbose: bool = True,
) -> dict:
    """把 qlib 本地数据从 start 续到 end（默认今天）。多进程并行拉。

    Args:
        start: 续期起始日（YYYY-MM-DD），默认 2020-09-28
        end: 续期结束日，默认今天
        universe: csi300/csi500/all，成分股列表
        instruments: 显式指定代码列表，覆盖 universe
        workers: 并行进程数
        verbose: 打进度
    Returns:
        {"days_added": int, "instruments_updated": int, "features_per_inst": int, ...}
    """
    if end is None:
        end = dt.date.today().isoformat()

    ensure_data_dirs()

    # 获取交易日
    new_days = _tencent_trading_days(start, end)
    if not new_days:
        return {"days_added": 0, "instruments_updated": 0, "features_per_inst": 0}
    if verbose:
        print(f"续期 {start} ~ {end}：{len(new_days)} 个交易日", flush=True)

    # 续日历文件
    cal_path = _QLIB_DATA_DIR / "calendars" / "day.txt"
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    existing_cal = cal_path.read_text().strip().split("\n") if cal_path.exists() else []
    new_cal_days = [d for d in new_days if d not in existing_cal]
    with open(cal_path, "a") as f:
        for d in new_cal_days:
            f.write(d + "\n")
    if verbose:
        print(f"日历续 {len(new_cal_days)} 日（去重后）", flush=True)

    # 拉成分股
    if instruments is None:
        instruments = _tencent_instruments(universe)
    if verbose:
        print(f"成分股 {len(instruments)} 只（universe={universe}），{workers} 进程并行", flush=True)

    # 多进程并行拉（带超时，防止卡死）
    feats_dir = _QLIB_DATA_DIR / "features"
    feats_dir.mkdir(parents=True, exist_ok=True)
    total = len(instruments)
    import time as _time
    import concurrent.futures as _cf

    _t0 = _time.time()
    ok_count = 0
    fail_count = 0
    done_count = 0
    # 每只股票单独提交任务，每任务超时 30 秒
    timeout_per_stock = 30

    with _cf.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for code in instruments:
            args = ([code], start, end, tuple(new_days), str(feats_dir), len(new_days))
            fut = executor.submit(_proc_batch, args)
            futures[fut] = code

        for fut in _cf.as_completed(futures, timeout=3600):
            code = futures[fut]
            done_count += 1
            try:
                ok_cnt, fail_cnt = fut.result(timeout=5)
                ok_count += ok_cnt
                fail_count += fail_cnt
            except Exception:
                fail_count += 1
                if verbose:
                    print(f"\n  ✗ {code} 超时或失败", flush=True)
            
            if verbose and (done_count % 50 == 0 or done_count == total):
                _elapsed = _time.time() - _t0
                _pct = min(done_count / total * 100, 100.0)
                _speed = done_count / _elapsed if _elapsed > 0 else 0
                print(f"\r  进度 {_pct:5.1f}%  ({done_count}/{total})  ✓{ok_count} ✗{fail_count}  已用 {_elapsed:.0f}s", end="", flush=True)

    if verbose:
        _total_elapsed = _time.time() - _t0
        print(f"\n完成：{ok_count} 只票 ✓，{fail_count} 只失败 ✗  × {len(_FEATURES)} 特征 × {len(new_days)} 日  ({_total_elapsed:.0f}s)", flush=True)

    # 生成 qlib instruments 文件（训练时必须）
    _generate_instruments(universe, instruments, verbose)

    return {
        "days_added": len(new_cal_days),
        "instruments_updated": ok_count,
        "features_per_inst": len(_FEATURES),
        "trading_days": len(new_days),
    }


def _generate_instruments(universe: str, instruments: list[str], verbose: bool = True) -> None:
    """生成 qlib 所需的 instruments 文件（如 csi300.txt）。

    qlib 格式：每行 code\\tstart_date\\tend_date（3 列 TAB 分隔，无表头）。
    """
    import datetime as dt
    end = dt.date.today().isoformat()
    start = "2026-01-01"
    inst_dir = _QLIB_DATA_DIR / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"{c.lower()}\t{start}\t{end}" for c in instruments]
    out_path = inst_dir / f"{universe}.txt"
    out_path.write_text("\n".join(lines))
    if verbose:
        print(f"  生成 instruments/{universe}.txt ({len(lines)} 只, TAB分隔)", flush=True)