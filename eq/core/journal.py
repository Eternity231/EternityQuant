"""纸面交易日志（v0.32 新增）—— 前向的、样本外的、没法作弊的验证。

**为什么必须有**

回测再漂亮也可能是过拟合的产物（v0.28 的 `eq bt optimize` 演示过：
样本内夏普 +0.57 → 样本外 -2.40）。唯一诚实的验证方式是：

    从今天起，把策略的每个推荐记下来 → 持有 N 个交易日后用真实行情结算
    → 和同期的沪深300 比超额收益。

记录之日起的表现没法作弊——没有窥探未来的可能，没有幸存者偏差，
没有"挑好看的区间"。攒够样本后 t 检验会告诉你：这是运气还是能力。

**核心指标是超额，不是绝对收益**

牛市里随便买都赚钱，不代表你会选股。真正要回答的问题是
「我的选股比直接买宽基 ETF 强吗」——所以每笔推荐都同时记下
基准（默认沪深300）同日价格，结算时算的是**超额收益**。

配套命令：``eq daily``（每天跑一次，自动记录+结算）、
``eq paper``（看战绩牌）。
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from typing import Any, Callable

import numpy as np
import pandas as pd

from eq.db import execute, get_state_conn

logger = logging.getLogger(__name__)

DEFAULT_BENCHMARK = "000300.SH"   # 沪深300：A 股散户最该比的基准
DEFAULT_HORIZON = 10              # 持有 10 个交易日再结算（约两周）


# ======================================================================
# 记录
# ======================================================================

def record(
    recos: list[dict[str, Any]],
    strategy: str,
    *,
    horizon_days: int = DEFAULT_HORIZON,
    benchmark: str = DEFAULT_BENCHMARK,
    benchmark_price: float | None = None,
) -> int:
    """记录一批推荐。同一天同一策略对同一标的只记一次（UNIQUE 去重）。

    Args:
        recos: ``[{"symbol", "date", "price", "note"?}, ...]``——
            ``date`` 是推荐日（该日收盘价为入场价），``price`` 是该日收盘价
        strategy: 策略名（战绩按策略分开统计）
        benchmark_price: 基准在推荐日的收盘价；None 表示记录时拿不到，
            结算时会退化为只算绝对收益

    Returns:
        实际新增条数（去重后）
    """
    if horizon_days <= 0:
        raise ValueError(f"horizon_days 必须为正：{horizon_days}")
    n = 0
    with get_state_conn() as conn:
        for r in recos:
            date = r["date"]
            if isinstance(date, (pd.Timestamp, dt.datetime)):
                date = date.date()
            price = float(r["price"])
            if price <= 0:
                logger.debug("跳过无效入场价 %s @ %s", r.get("symbol"), price)
                continue
            cur = conn.execute(
                """INSERT OR IGNORE INTO paper_recos
                   (reco_date, symbol, strategy, entry_price, horizon_days,
                    benchmark, benchmark_entry, note)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (date.isoformat() if hasattr(date, "isoformat") else str(date),
                 r["symbol"], strategy, price, int(horizon_days),
                 benchmark, benchmark_price, r.get("note")),
            )
            n += cur.rowcount
        conn.commit()
    return n


# ======================================================================
# 结算
# ======================================================================

def _exit_after(bars: pd.DataFrame, reco_date, horizon: int) -> tuple[Any, float] | None:
    """在 ``reco_date`` 之后的第 ``horizon`` 个交易日收盘结算。

    交易日不足 ``horizon`` 个 → 还没到期，返回 None。
    用「行情里的实际交易日」数，而不是自然日换算——节假日、停牌都自动对。
    """
    after = bars[bars.index > pd.Timestamp(reco_date)]
    if len(after) < horizon:
        return None
    row = after.iloc[horizon - 1]
    return after.index[horizon - 1], float(row["close"])


def evaluate_due(
    fetch: Callable[[str, int], pd.DataFrame] | None = None,
    *,
    max_rows: int = 500,
) -> list[dict[str, Any]]:
    """把已到期的 open 推荐结算掉。返回本次结算的明细。

    Args:
        fetch: ``fetch(symbol, days) -> bars``，默认走
            :func:`eq.data.market.get_recent_bars`（带缓存）。测试时注入假数据。
    """
    if fetch is None:
        from eq.data.market import get_recent_bars

        def fetch(sym: str, days: int) -> pd.DataFrame:   # type: ignore[misc]
            return get_recent_bars(sym, days=days)

    rows = execute(
        "SELECT id, reco_date, symbol, strategy, entry_price, horizon_days, "
        "benchmark, benchmark_entry FROM paper_recos WHERE status = 'open' "
        "ORDER BY reco_date LIMIT ?",
        (max_rows,),
    )
    if not rows:
        return []

    # 行情按标的取一次；基准按名字取一次
    bars_cache: dict[str, pd.DataFrame | None] = {}

    def _bars(sym: str, horizon: int) -> pd.DataFrame | None:
        if sym not in bars_cache:
            try:
                bars_cache[sym] = fetch(sym, max(horizon * 3 + 40, 90))
            except Exception as e:
                logger.debug("结算取数失败 %s：%s", sym, e)
                bars_cache[sym] = None
        return bars_cache[sym]

    settled: list[dict[str, Any]] = []
    with get_state_conn() as conn:
        for r in rows:
            d = dict(r)
            horizon = int(d["horizon_days"])
            bars = _bars(d["symbol"], horizon)
            if bars is None or bars.empty:
                continue
            hit = _exit_after(bars, d["reco_date"], horizon)
            if hit is None:
                continue    # 还没到期
            exit_date, exit_px = hit

            bench_exit = None
            if d["benchmark"] and d["benchmark_entry"]:
                bb = _bars(d["benchmark"], horizon)
                if bb is not None and not bb.empty:
                    bh = _exit_after(bb, d["reco_date"], horizon)
                    if bh is not None:
                        bench_exit = bh[1]

            conn.execute(
                """UPDATE paper_recos SET status='closed', exit_date=?,
                   exit_price=?, benchmark_exit=? WHERE id=?""",
                (pd.Timestamp(exit_date).date().isoformat(), exit_px, bench_exit, d["id"]),
            )
            ret = exit_px / d["entry_price"] - 1
            bench_ret = (bench_exit / d["benchmark_entry"] - 1
                         if bench_exit and d["benchmark_entry"] else None)
            settled.append({
                "symbol": d["symbol"], "strategy": d["strategy"],
                "reco_date": d["reco_date"], "exit_date": str(pd.Timestamp(exit_date).date()),
                "ret": float(ret),
                "bench_ret": float(bench_ret) if bench_ret is not None else None,
                "excess": float(ret - bench_ret) if bench_ret is not None else None,
            })
        conn.commit()
    return settled


# ======================================================================
# 战绩牌
# ======================================================================

def scoreboard(strategy: str | None = None) -> dict[str, Any]:
    """汇总纸面战绩。核心是**超额收益的 t 统计量**——回答"运气还是能力"。"""
    q = ("SELECT reco_date, symbol, strategy, entry_price, exit_price, "
         "benchmark_entry, benchmark_exit, horizon_days FROM paper_recos "
         "WHERE status = 'closed'")
    params: tuple = ()
    if strategy:
        q += " AND strategy = ?"
        params = (strategy,)
    closed = [dict(r) for r in execute(q, params)]

    q2 = "SELECT COUNT(*) c FROM paper_recos WHERE status = 'open'"
    if strategy:
        q2 += " AND strategy = ?"
    n_open = execute(q2, params)[0]["c"]

    if not closed:
        return {"n_closed": 0, "n_open": int(n_open), "strategy": strategy or "全部"}

    rets = np.array([r["exit_price"] / r["entry_price"] - 1 for r in closed])
    excess = np.array([
        (r["exit_price"] / r["entry_price"]) - (r["benchmark_exit"] / r["benchmark_entry"])
        for r in closed
        if r["benchmark_exit"] and r["benchmark_entry"]
    ])

    out: dict[str, Any] = {
        "strategy": strategy or "全部",
        "n_closed": len(closed),
        "n_open": int(n_open),
        "ret_mean": float(rets.mean()),
        "ret_median": float(np.median(rets)),
        "win_rate": float((rets > 0).mean()),
    }
    if len(excess):
        mean, std = float(excess.mean()), float(excess.std(ddof=1)) if len(excess) > 1 else 0.0
        t = mean / std * math.sqrt(len(excess)) if std > 0 else 0.0
        out.update({
            "n_vs_bench": len(excess),
            "excess_mean": mean,
            "excess_median": float(np.median(excess)),
            "beat_bench_rate": float((excess > 0).mean()),
            "excess_t": float(t),
            # 按当前均值/波动，还需要多少笔才能到 t=2（统计显著）
            "n_for_significance": (int(math.ceil((2 * std / mean) ** 2))
                                   if mean > 0 and std > 0 else None),
        })
    return out


def recent_closed(limit: int = 20, strategy: str | None = None) -> list[dict[str, Any]]:
    """最近结算的明细，复盘用。"""
    q = ("SELECT reco_date, symbol, strategy, entry_price, exit_price, exit_date, "
         "benchmark_entry, benchmark_exit FROM paper_recos WHERE status='closed'")
    params: list[Any] = []
    if strategy:
        q += " AND strategy = ?"
        params.append(strategy)
    q += " ORDER BY exit_date DESC, id DESC LIMIT ?"
    params.append(limit)
    out = []
    for r in execute(q, tuple(params)):
        d = dict(r)
        d["ret"] = d["exit_price"] / d["entry_price"] - 1
        d["excess"] = (d["ret"] - (d["benchmark_exit"] / d["benchmark_entry"] - 1)
                       if d["benchmark_exit"] and d["benchmark_entry"] else None)
        out.append(d)
    return out


def format_scoreboard(sb: dict[str, Any]) -> str:
    """战绩牌排版。重点是把「显著不显著」讲成人话。"""
    lines = [f"\n纸面战绩（{sb['strategy']}）"]
    if not sb.get("n_closed"):
        lines.append(f"  还没有已结算的推荐（在途 {sb.get('n_open', 0)} 笔）。")
        lines.append("  每天跑一次 `eq daily`，推荐会自动记录并在到期后结算。")
        return "\n".join(lines) + "\n"
    lines.append(
        f"  已结算 {sb['n_closed']} 笔   在途 {sb['n_open']} 笔   "
        f"胜率 {sb['win_rate']:.0%}   平均收益 {sb['ret_mean']:+.2%}"
        f"（中位 {sb['ret_median']:+.2%}）"
    )
    if sb.get("n_vs_bench"):
        t = sb["excess_t"]
        lines.append(
            f"  vs 基准：超额均值 {sb['excess_mean']:+.2%}"
            f"（中位 {sb['excess_median']:+.2%}）   跑赢占比 {sb['beat_bench_rate']:.0%}"
            f"   t={t:+.2f}"
        )
        if abs(t) >= 2:
            verdict = ("统计上显著地跑赢基准——继续保持" if t > 0
                       else "统计上显著地**跑输**基准——该停下来反思选股方法了")
        elif sb["excess_mean"] > 0:
            need = sb.get("n_for_significance")
            verdict = ("暂时领先但不显著（|t|<2），继续攒样本"
                       + (f"——按当前水平约需 {need} 笔" if need and need < 10000 else ""))
        else:
            verdict = "暂时落后于基准。样本还少，先别下结论，但也别加仓"
        lines.append(f"  判定：{verdict}")
    return "\n".join(lines) + "\n"
