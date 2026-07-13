"""回测结果外存 + metadata 入 backtest_runs 表（problem 15 冶议）。

SQLite 只存 metadata（策略名、指标、时间），详细数据（逐日权益、交易明细）写到
~/.eternityquant/backtests/<run_id>.parquet，避免 SQLite 存大数据的瓶颈。
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from eq.backtest.types import BacktestResult
from eq.db import DEFAULT_HOME, execute, execute_write, get_state_conn

_BACKTESTS_DIR = DEFAULT_HOME / "backtests"


def _ensure_dir() -> Path:
    _BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)
    return _BACKTESTS_DIR


def save_result(
    result: BacktestResult,
    symbol: str,
    strategy_name: str,
) -> str:
    """把回测结果外存 parquet + metadata 入 backtest_runs 表，返回 run_id。

    parquet 内含两个 sheet：equity_curve（index=date, value=equity）和 trades（明细）。
    """
    run_id = f"bt_{dt.date.today().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}"
    artifact = _ensure_dir() / f"{run_id}.parquet"

    # 写 parquet：用 dict 多 sheet 模式（pandas to_parquet 不支持多 sheet，改用两类拼接 + 独立索引）
    # 简化第一版：写两个文件——<id>.equity.parquet 和 <id>.trades.parquet，metadata 里记两者
    equity = result.equity_curve.to_frame(name="equity")
    trades = result.trades.copy() if not result.trades.empty else pd.DataFrame()
    equity.to_parquet(artifact.with_suffix(".equity.parquet"))
    trades.to_parquet(artifact.with_suffix(".trades.parquet"))

    # metadata 入表
    cfg = result.config
    execute_write(
        """INSERT INTO backtest_runs (id, symbol, strategy_name, engine, config, metrics, artifact_path)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id, symbol, strategy_name, cfg.engine,
            json.dumps({
                "initial_cash": cfg.initial_cash,
                "commission_bps": cfg.commission_bps,
                "slippage_bps": cfg.slippage_bps,
                "allow_short": cfg.allow_short,
            }, ensure_ascii=False),
            json.dumps(result.metrics, ensure_ascii=False),
            str(artifact),
        ),
    )
    return run_id


def list_runs(symbol: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    """列出最近 N 个回测记录，可选按标的过滤。"""
    q = "SELECT id, symbol, strategy_name, engine, metrics, created_at FROM backtest_runs"
    params: tuple = ()
    if symbol:
        q += " WHERE symbol = ?"
        params = (symbol,)
    q += " ORDER BY created_at DESC LIMIT ?"
    params = params + (limit,)
    rows = execute(q, params)
    out = []
    for r in rows:
        d = {k: r[k] for k in r.keys()}
        d["metrics"] = json.loads(d["metrics"] or "{}")
        out.append(d)
    return out


def load_result(run_id: str) -> dict[str, Any]:
    """按 run_id 加载完整回测结果（metadata + equity + trades）。"""
    rows = execute("SELECT * FROM backtest_runs WHERE id = ?", (run_id,))
    if not rows:
        raise KeyError(f"回测记录 {run_id} 不存在")
    meta = {k: rows[0][k] for k in rows[0].keys()}
    meta["config"] = json.loads(meta["config"] or "{}")
    meta["metrics"] = json.loads(meta["metrics"] or "{}")
    artifact = Path(meta["artifact_path"])
    equity = pd.read_parquet(artifact.with_suffix(".equity.parquet")) if artifact.with_suffix(".equity.parquet").exists() else pd.DataFrame()
    trades = pd.read_parquet(artifact.with_suffix(".trades.parquet")) if artifact.with_suffix(".trades.parquet").exists() else pd.DataFrame()
    return {"meta": meta, "equity": equity, "trades": trades}


def remove_run(run_id: str) -> bool:
    """删除回测记录：SQLite metadata + parquet 文件。"""
    rows = execute("SELECT artifact_path FROM backtest_runs WHERE id = ?", (run_id,))
    if not rows:
        return False
    artifact = Path(rows[0]["artifact_path"])
    for suffix in [".equity.parquet", ".trades.parquet"]:
        p = artifact.with_suffix(suffix)
        if p.exists():
            p.unlink()
    with get_state_conn() as conn:
        conn.execute("DELETE FROM backtest_runs WHERE id = ?", (run_id,))
        conn.commit()
    return True
