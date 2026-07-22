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

class _TencentResult:
    """腾讯 API 单次请求结果，区分「未上市/退市」和「网络错误」。

    - data 非 None：正常 DataFrame（可能为空，表示该区间无数据）
    - is_empty_range=True：该股票在 [start, end] 区间无任何数据，
      常见原因是未上市/已退市/区间在上市前。调用方**不应重试**，
      应直接写全 NaN 跳过。
    - is_network_error=True：网络/API 故障，调用方可重试。
    """

    __slots__ = ("data", "is_empty_range", "is_network_error")

    def __init__(self, data, *, is_empty_range: bool = False, is_network_error: bool = False):
        self.data = data
        self.is_empty_range = is_empty_range
        self.is_network_error = is_network_error


def _tencent_daily(code: str, start: str, end: str) -> _TencentResult:
    """从腾讯 API 拉单只股票日线，返回 :class:`_TencentResult`。

    Args:
        code: qlib 格式如 'SH600000' 或 'SZ000001'
        start: YYYY-MM-DD
        end: YYYY-MM-DD
    Returns:
        _TencentResult；``data`` 为 DataFrame(index=date) 或 None
    """
    # 转腾讯格式：sh600000 / sz000001
    if code.startswith("SH"):
        tcode = "sh" + code[2:]
    elif code.startswith("SZ"):
        tcode = "sz" + code[2:]
    else:
        return _TencentResult(None, is_empty_range=True)

    try:
        url = f"{_TENCENT_KLINE}?param={tcode},day,{start},{end},640,qfq"
        resp = _req.get(url, timeout=15)
        if resp.status_code != 200:
            return _TencentResult(None, is_network_error=True)
        data = resp.json()
        if data.get("code") != 0:
            # 腾讯对未上市/已退市股票返回非 0 code，视为「无数据」而非网络错误
            return _TencentResult(None, is_empty_range=True)
        # 腾讯返回 "qfqday"（前复权）或 "day"（不复权）
        stock_data = data.get("data", {}).get(tcode, {})
        bars = stock_data.get("qfqday") or stock_data.get("day")
        if not bars:
            # 正常返回但无任何 bar —— 该股票在 [start, end] 内确实无数据
            return _TencentResult(None, is_empty_range=True)
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
            return _TencentResult(None, is_empty_range=True)
        df = pd.DataFrame(rows).set_index("date")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return _TencentResult(df)
    except (_req.RequestException, _req.Timeout, ValueError) as e:
        # 网络异常或 JSON 解析失败 —— 可重试
        print(f"  [DEBUG] {code} 网络异常: {e}", flush=True)
        return _TencentResult(None, is_network_error=True)
    except Exception as e:
        print(f"  [DEBUG] {code} 异常: {e}", flush=True)
        return _TencentResult(None, is_network_error=True)


def _watchlist_instruments() -> list[str]:
    """从本地自选股文件提取 A 股代码。

    读取 ``D:\\idmxz\\Table.txt``（Tab 分隔，第一列是代码），
    提取 A 股代码（SH/SZ/BJ 前缀，或裸 6 位数字码），
    转为 qlib 格式 ``["SH600000", "SZ000001"]``。

    文件不存在或读取失败时回退到已有 features/ 目录扫描。
    """
    codes: list[str] = []
    table_path = Path("D:/idmxz/Table.txt")
    if not table_path.exists():
        print(f"  [warn] 自选股文件不存在: {table_path}", flush=True)
        return codes

    try:
        raw = table_path.read_text(encoding="utf-8").splitlines()
        for line in raw:
            line = line.strip()
            if not line or line.startswith("代码") or "\t" not in line:
                continue
            code = line.split("\t")[0].strip()
            if not code:
                continue
            # 已有前缀
            if code.startswith(("SH", "SZ", "BJ")):
                codes.append(code.upper())
            # 裸 6 位数字 → 推断交易所
            elif code.isdigit() and len(code) == 6:
                if code.startswith(("6", "9")):
                    codes.append(f"SH{code}")
                elif code.startswith(("0", "3")):
                    codes.append(f"SZ{code}")
                elif code.startswith(("4", "8")):
                    codes.append(f"BJ{code}")
            # 其他（港股/美股/指数等）跳过
    except Exception as e:
        print(f"  [warn] 读取自选股文件失败: {e}", flush=True)
        return codes

    # 去重
    seen = set()
    deduped = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


def _tencent_instruments(universe: str = "csi300") -> list[str]:
    """获取成分股列表。

    优先级：
    1. 本地缓存 ``{universe}_codes.txt``（逗号分隔的 qlib 代码如 ``SH600000``）
    2. akshare ``index_stock_cons_csindex``（中证指数公司官方源，最稳定）
    3. akshare ``index_stock_cons``（新浪源，老版 akshare 可用）
    4. 全 A 股：``stock_zh_a_spot``
    5. 自选股：``watchlist``（从 ``D:\\idmxz\\Table.txt`` 读取）
    6. fallback：从已有 features/ 目录扫描

    Args:
        universe: csi300 | csi500 | sz50 | all | watchlist
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

    # 5. 自选股（从 D:\idmxz\Table.txt 读取）
    if universe == "watchlist":
        codes = _watchlist_instruments()

    # 6. 已有 features/ 目录扫描
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

    覆盖写策略：每只股票在 [start, end] 内**整段重算**并覆盖 .bin，
    不再追加。因此多次下载（如先 2016 再 2026，或反之）得到完全
    相同的 .bin 内容，结果与下载顺序无关。

    未上市/已退市处理：腾讯返回空数据时，判定为该区间无数据，
    **直接跳过不重试**；.bin 文件写全 NaN（按交易日历对齐），
    qlib instruments 文件中可用区间为空，训练时自动忽略。

    Args:
        code: 如 "SH600000"
        start, end: 日期范围
        new_days: 交易日列表
        qlib_feats_dir: 特征输出目录
        expected_floats: 每个 .bin 文件预期的 float32 数量（用于幂等检查）
    Returns:
        bool: 是否成功
    """
    import time as _t
    from pathlib import Path

    inst_dir = Path(qlib_feats_dir) / code.lower()
    inst_dir.mkdir(parents=True, exist_ok=True)
    days_list = list(new_days)
    n_days = len(days_list)

    # 完整性检测：所有 7 个特征 .bin 都必须存在且长度精确等于交易日数，
    # 才视为「已完整下载」可跳过。任一缺失/长度不符/全 NaN（未上市）→ 视为
    # 损坏或不完整，覆盖重下。这样能抓到中途崩溃留下的半截 .bin。
    if expected_floats > 0:
        all_complete = True
        for feat in _FEATURES:
            bp = inst_dir / f"{feat}.day.bin"
            if not bp.exists() or bp.stat().st_size // 4 != expected_floats:
                all_complete = False
                break
        if all_complete:
            return True

    # 网络错误最多重试 3 次（指数退避）；
    # 未上市/已退市返回 is_empty_range=True，直接写全 NaN 跳过。
    # 首请不 sleep（腾讯 API 国内直连限流宽松，靠进程间并发即可），
    # 重试时才指数退避 2s/4s。
    #
    # 「--start 早于上市日」处理：腾讯对 [start, end] 全段返回空时，
    # 不能直接判"未上市"——可能是 start 早于上市日但上市日在 [start, end] 内。
    # 此时按年扩窗探测：从 start 起每次把 start 往后推 1 年重试，
    # 直到拿到数据（说明上市日找到了）或撞到 end（真未上市/已退市）。
    df = None
    is_empty_range = False
    probe_start = start
    for attempt in range(3):
        if attempt > 0:
            _t.sleep(2 ** attempt)  # 指数退避 2s, 4s
        result = _tencent_daily(code, probe_start, end)
        if result.is_network_error:
            # 网络故障，重试
            if attempt >= 2:
                print(f"  [DEBUG] {code} 网络错误，3 次重试仍失败，跳过", flush=True)
            continue
        if result.is_empty_range:
            # [probe_start, end] 全空。若 probe_start 已 == start，
            # 先尝试按年往后推探测上市日；若推到 ≥ end 仍空 → 真未上市。
            import datetime as _dt_mod
            try:
                ps = _dt_mod.date.fromisoformat(probe_start)
            except ValueError:
                is_empty_range = True
                break
            next_ps = ps.replace(year=ps.year + 1)
            if next_ps.isoformat() <= end:
                probe_start = next_ps.isoformat()
                _t.sleep(0.3)  # 跨年探测间隔，避免连发多年请求触发限流
                continue  # 同 attempt 内继续探测，不耗重试次数
            # 推到 ≥ end 仍空 → 真未上市/已退市
            is_empty_range = True
            break
        df = result.data
        break

    if is_empty_range:
        # 该股票在 [start, end] 内无数据，按交易日历写全 NaN，
        # 便于后续覆盖重写与对齐。这属于「正常跳过」，算成功处理。
        for feat in _FEATURES:
            _write_bin(inst_dir / f"{feat}.day.bin", [float("nan")] * n_days)
        return True
    elif df is None or df.empty:
        # 网络重试耗尽 —— 真失败，不写 .bin（保持旧文件不变）
        return False
    else:
        # 按交易日对齐：reindex 后未上市/停牌日为 NaN。
        # 若 probe_start > start，则上市日之前的交易日自然落 NaN，
        # 后续 ffill/bfill 不会污染（NaN 仍是 NaN，直到首个有效日）。
        df = df.reindex(pd.to_datetime(days_list))
        # 前向填充 NaN（停牌日沿用上一交易日数据）
        df = df.ffill()
        # 如果开头还有 NaN（上市前的交易日），用后续第一个有效值填充，
        # 这样上市前的 NaN 也被填成首日开盘价附近，避免 qlib 特征计算报错。
        # 但全 NaN（仍未上市）时 bfill 也无效，下面统一处理。
        df = df.bfill()

    if df["close"].isna().all():
        # 整段无数据，写全 NaN 的 .bin（保持文件存在与长度一致）
        for feat in _FEATURES:
            _write_bin(inst_dir / f"{feat}.day.bin", [float("nan")] * n_days)
        return True

    df["factor"] = 1.0
    df["change"] = df["close"].pct_change(fill_method=None).fillna(0.0)
    # 转 .bin（覆盖写，保证顺序无关）
    for feat in _FEATURES:
        if feat in df.columns:
            vals = df[feat].tolist()
        else:
            vals = [float("nan")] * n_days
        vals = [float("nan") if (v != v or v is None) else v for v in vals]
        _write_bin(inst_dir / f"{feat}.day.bin", vals)
    return True


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


def _write_bin(bin_path: Path, values: list[float]) -> None:
    """覆盖写 .bin 文件（整段重算，结果与下载顺序无关）。

    与 :func:`_append_bin` 的区别：本函数用 ``"wb"`` 整段写，
    保证多次下载同一区间得到完全相同的 .bin 内容。
    """
    arr = np.array(values, dtype=np.float32)
    with open(bin_path, "wb") as f:
        f.write(arr.tobytes())


# ---------- 主入口 ----------

def update_qlib_data(
    start: str = "2020-09-28",
    end: str | None = None,
    universe: str = "csi300",
    instruments: list[str] | None = None,
    extra_codes: list[str] | None = None,
    workers: int = 5,  # 默认 5 进程，避免 API 限流和 CPU 爆满
    verbose: bool = True,
) -> dict:
    """把 qlib 本地数据从 start 续到 end（默认今天）。多进程并行拉。

    「单只股票 + 预设指数共同下载并训练」：传 ``extra_codes`` 额外指定
    一批股票代码（如 ``["SH600519"]``），它们会与 ``universe``（如
    csi500）的成分股**合并、去重**后一起下载，并写进同一份 instruments
    文件，随后可用于训练。

    「下载先后顺序无关」：每次更新 [start, end] 区间都会**整段重算并
    覆盖**该区间内每只股票的 .bin 文件；calendars/day.txt 与
    instruments/<universe>.txt 也都是基于完整交易日列表、按固定顺序
    生成后覆盖写。因此无论先下 2016 还是先下 2026，最终 .bin 与
    instruments 文件内容完全一致。

    「跳过较晚股票没上市的时间不重试」：腾讯返回空数据（未上市/已退
    市/区间在上市前）时，直接判定为该区间无数据，**不重试**，.bin
    写全 NaN；instruments 文件中该股票可用区间为空，训练时自动忽略。

    Args:
        start: 续期起始日（YYYY-MM-DD），默认 2020-09-28
        end: 续期结束日，默认今天
        universe: csi300/csi500/all，成分股列表
        instruments: 显式指定代码列表，覆盖 universe
        extra_codes: 额外股票代码列表（如 ["SH600519"]），与 universe
            合并去重后一起下载训练
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

    # 日历文件：去重 + 升序，覆盖写（结果与下载顺序无关）
    cal_path = _QLIB_DATA_DIR / "calendars" / "day.txt"
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    existing_cal = cal_path.read_text().strip().split("\n") if cal_path.exists() else []
    all_cal_days = sorted(set(existing_cal) | set(new_days))
    cal_path.write_text("\n".join(all_cal_days) + "\n")
    new_cal_days = [d for d in new_days if d not in existing_cal]
    if verbose:
        print(f"日历合并后 {len(all_cal_days)} 日，本次新增 {len(new_cal_days)} 日", flush=True)

    # 拉成分股（universe） + 额外指定的单只股票，合并去重
    if instruments is None:
        instruments = _tencent_instruments(universe)
    extra_codes = [c.upper().strip() for c in (extra_codes or []) if c and c.strip()]
    merged = list(instruments)
    for c in extra_codes:
        if c not in merged:
            merged.append(c)
    if verbose:
        extra_part = f"，含额外 {len(extra_codes)} 只" if extra_codes else ""
        print(
            f"成分股 {len(instruments)} 只（universe={universe}）{extra_part}"
            f"，合并去重后 {len(merged)} 只，{workers} 进程并行",
            flush=True,
        )

    # 多进程并行拉：把 merged 切成 workers 块，每进程串行处理一块，
    # 比「每只一 future」少 N 倍进程间调度开销，且腾讯 API 单 TCP 串行无影响。
    feats_dir = _QLIB_DATA_DIR / "features"
    feats_dir.mkdir(parents=True, exist_ok=True)
    total = len(merged)
    import time as _time
    import concurrent.futures as _cf

    # 切块：尽量均匀，最后一块兜底
    chunks: list[list[str]] = [[] for _ in range(max(workers, 1))]
    for i, code in enumerate(merged):
        chunks[i % len(chunks)].append(code)
    chunks = [c for c in chunks if c]  # 丢空块

    _t0 = _time.time()
    ok_count = 0
    fail_count = 0
    done_count = 0

    with _cf.ProcessPoolExecutor(max_workers=max(workers, 1)) as executor:
        futures = {}
        for chunk in chunks:
            args = (chunk, start, end, tuple(new_days), str(feats_dir), len(new_days))
            fut = executor.submit(_proc_batch, args)
            futures[fut] = len(chunk)

        for fut in _cf.as_completed(futures, timeout=3600):
            chunk_size = futures[fut]
            done_count += chunk_size
            try:
                ok_cnt, fail_cnt = fut.result(timeout=30)
                ok_count += ok_cnt
                fail_count += fail_cnt
            except Exception:
                fail_count += chunk_size
                if verbose:
                    print(f"\n  ✗ 一块（{chunk_size} 只）超时或失败", flush=True)

            if verbose and (done_count % 50 == 0 or done_count >= total):
                _elapsed = _time.time() - _t0
                _pct = min(done_count / total * 100, 100.0)
                _speed = done_count / _elapsed if _elapsed > 0 else 0
                print(f"\r  进度 {_pct:5.1f}%  ({done_count}/{total})  ✓{ok_count} ✗{fail_count}  已用 {_elapsed:.0f}s", end="", flush=True)

    if verbose:
        _total_elapsed = _time.time() - _t0
        print(f"\n完成：{ok_count} 只票 ✓，{fail_count} 只失败 ✗  × {len(_FEATURES)} 特征 × {len(new_days)} 日  ({_total_elapsed:.0f}s)", flush=True)

    # 生成 qlib instruments 文件（训练时必须）
    # 用合并后的 merged（含 extra_codes），保证训练池与下载池一致
    _generate_instruments(universe, merged, verbose)

    return {
        "days_added": len(new_cal_days),
        "instruments_updated": ok_count,
        "features_per_inst": len(_FEATURES),
        "trading_days": len(new_days),
    }


def _generate_instruments(universe: str, instruments: list[str], verbose: bool = True) -> None:
    """生成 qlib 所需的 instruments 文件（如 csi300.txt）。

    qlib 格式：每行 ``code\\tstart_date\\tend_date``（3 列 TAB 分隔，无表头）。

    每只股票的 start_date / end_date 由其 ``close.day.bin`` 的实际数据
    推断：扫描 .bin 找首/尾非 NaN 的索引，再映射到 calendars/day.txt
    的对应交易日。因此 instruments 文件内容**只取决于 .bin 与日历**，
    与下载顺序无关。

    未上市/已退市的股票 .bin 全 NaN，扫描后 in_market = False，
    该行用 ``instruments`` 列表里的原始占位区间（start=2000-01-01，
    end=今天）—— qlib 训练时会因无数据自动忽略。占位区间设得足够宽
    （2000~today），覆盖任何训练区间，避免 list_instruments 滤空。
    """
    import datetime as dt
    import numpy as _np

    today = dt.date.today().isoformat()
    default_start = "2000-01-01"

    # 读日历，用于把 .bin 索引映射回日期
    cal_path = _QLIB_DATA_DIR / "calendars" / "day.txt"
    cal_days: list[str] = []
    if cal_path.exists():
        cal_days = [ln.strip() for ln in cal_path.read_text().splitlines() if ln.strip()]

    feats_dir = _QLIB_DATA_DIR / "features"
    inst_dir = _QLIB_DATA_DIR / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for code in instruments:
        c = code.lower()
        close_bin = feats_dir / c / "close.day.bin"
        start_d = default_start
        end_d = today
        if close_bin.exists() and close_bin.stat().st_size >= 4 and cal_days:
            arr = _np.fromfile(close_bin, dtype=_np.float32)
            # 找首/尾非 NaN 索引
            non_nan_mask = ~_np.isnan(arr)
            if non_nan_mask.any():
                first_idx = int(_np.argmax(non_nan_mask))
                last_idx = int(len(arr) - 1 - _np.argmax(non_nan_mask[::-1]))
                # 映射到日历日期（.bin 索引 = 日历索引，前提是 .bin 是按
                # 全日历对齐写的；本模块 _proc_one_stock 用 new_days 对齐，
                # new_days ⊆ 全日历，索引一致）
                if first_idx < len(cal_days):
                    start_d = cal_days[first_idx]
                if last_idx < len(cal_days):
                    end_d = cal_days[last_idx]
        lines.append(f"{c}\t{start_d}\t{end_d}")

    out_path = inst_dir / f"{universe}.txt"
    out_path.write_text("\n".join(lines) + "\n")
    if verbose:
        print(f"  生成 instruments/{universe}.txt ({len(lines)} 只, TAB分隔)", flush=True)