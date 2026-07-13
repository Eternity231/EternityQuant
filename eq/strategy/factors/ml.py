"""ML 因子包装：qlib 预测输出作为因子喂给信号层（problem 16 冶议）。

架构原则（problem 8）：qlib 是外部数据源，EternityQuant 自控信号决策。
ML 因子从 ml_predictions 表读最近一天的预测分数列，转成因子数列。
qlib 挂了 → ML 因子返回 NaN，框架照跑（非硬依赖）。
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from typing import Any

import pandas as pd

from eq.db import execute, get_state_conn


def _active_model_id(universe: str) -> str | None:
    """查指定 universe 当前激活的模型 id。无则返回 None。"""
    rows = execute("SELECT id FROM ml_models WHERE universe = ? AND is_active = 1 LIMIT 1", (universe,))
    return rows[0]["id"] if rows else None


def ml_score(df: pd.DataFrame, universe: str = "csi300", horizon: int | None = None) -> pd.Series:
    """qlib 预测分数因子。

    Args:
        df: 价格 DataFrame，index 为日期，用于对齐预测列到价格日期
        universe: 标的池名，查该池的激活模型
        horizon: 可选，筛指定预测窗口的模型

    Returns:
        pd.Series，index 同 df，值为 qlib 预测分数（NaN 表示无预测或模型挂了）
    """
    model_id = _active_model_id(universe)
    if model_id is None:
        return pd.Series(pd.NA, index=df.index, name="ml_score")
    # 拉该模型所有预测，按 symbol/date 索引
    rows = execute(
        "SELECT symbol, date, score FROM ml_predictions WHERE model_id = ?",
        (model_id,),
    )
    if not rows:
        return pd.Series(pd.NA, index=df.index, name="ml_score")
    pred_df = pd.DataFrame([dict(r) for r in rows])
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    # 第一版简化：取最近一天的预测分数，对所有标的均匀广播到价格 df 的每个日期
    # 真实场景：每个标的都有独立 ml_score 数列，需要按标的分组。本函数在被单标的调用时适用
    latest = pred_df.sort_values("date").iloc[-1]
    return pd.Series(float(latest["score"]), index=df.index, name="ml_score")


def register_model(
    name: str,
    universe: str,
    features: list[str],
    algo: str,
    horizon: int,
    train_period: str,
    valid_period: str = "",
    metrics: dict[str, Any] | None = None,
    model_path: str = "",
    notes: str = "",
) -> str:
    """登记一个新训练完成的模型，返回 model_id（UUID）。不自动激活。"""
    import uuid as _uuid
    model_id = f"m_{dt.date.today().strftime('%Y%m%d')}_{_uuid.uuid4().hex[:6]}"
    from eq.db import execute_write
    execute_write(
        """INSERT INTO ml_models
           (id, name, universe, features, algo, horizon, train_period, valid_period, metrics, model_path, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            model_id, name, universe, json.dumps(features, ensure_ascii=False),
            algo, horizon, train_period, valid_period,
            json.dumps(metrics or {}, ensure_ascii=False), model_path, notes or None,
        ),
    )
    return model_id


def activate(model_id: str) -> bool:
    """激活某模型（同 universe 其他模型自动停用）。"""
    with get_state_conn() as conn:
        # 查 universe
        row = conn.execute("SELECT universe FROM ml_models WHERE id = ?", (model_id,)).fetchone()
        if row is None:
            return False
        universe = row["universe"]
        conn.execute("UPDATE ml_models SET is_active = 0 WHERE universe = ?", (universe,))
        cur = conn.execute("UPDATE ml_models SET is_active = 1 WHERE id = ?", (model_id,))
        conn.commit()
        return cur.rowcount > 0


def list_models(universe: str | None = None) -> list[dict[str, Any]]:
    """列出模型，按训练时间倒序。可选按 universe 过滤。"""
    q = "SELECT id, name, universe, algo, horizon, trained_at, train_period, is_active, metrics, notes FROM ml_models"
    params: tuple = ()
    if universe:
        q += " WHERE universe = ?"
        params = (universe,)
    q += " ORDER BY trained_at DESC"
    rows = execute(q, params)
    out = []
    for r in rows:
        d = {k: r[k] for k in r.keys()}
        d["metrics"] = json.loads(d["metrics"] or "{}")
        out.append(d)
    return out


def save_prediction(model_id: str, symbol: str, date: dt.date, score: float) -> None:
    """写入一条预测。因子层调 ml_score() 时读这表。"""
    from eq.db import execute_write
    execute_write(
        "INSERT INTO ml_predictions (model_id, symbol, date, score) VALUES (?, ?, ?, ?)",
        (model_id, symbol, date.isoformat(), score),
    )
