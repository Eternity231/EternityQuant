"""SQLite 数据访问层。

两个数据库文件：
- eternityquant.db：状态库（watchlist/portfolio/trade_history/rules/signals）
- market_cache.db：行情缓存（可随时删）

所有表 schema 在首次连接时通过 CREATE TABLE IF NOT EXISTS 幂等建立。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_HOME = Path.home() / ".eternityquant"

_SCHEMA_STATE = """
CREATE TABLE IF NOT EXISTS watchlist (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol     TEXT    UNIQUE NOT NULL,
    name       TEXT,
    market     TEXT,
    added_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reason     TEXT,
    tags       TEXT
);

CREATE TABLE IF NOT EXISTS portfolio (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT UNIQUE NOT NULL,
    name          TEXT,
    market        TEXT,
    shares        REAL     NOT NULL DEFAULT 0,
    cost_price    REAL     NOT NULL DEFAULT 0,
    opened_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    stop_loss     REAL,
    take_profit   REAL,
    status        TEXT     NOT NULL DEFAULT 'open',   -- open / closed
    closed_at     TIMESTAMP,
    realized_pnl  REAL     NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS trade_history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT    NOT NULL,
    action        TEXT    NOT NULL,   -- buy / sell / add / trim
    shares        REAL    NOT NULL,
    price         REAL    NOT NULL,
    executed_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    note          TEXT
);

CREATE INDEX IF NOT EXISTS idx_trade_history_symbol ON trade_history(symbol);

CREATE TABLE IF NOT EXISTS rules (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT,                              -- NULL = 全市场规则
    type          TEXT    NOT NULL,                  -- 规则类型枚举（problem 11）
    params        TEXT,                              -- JSON 参数
    channels      TEXT    NOT NULL DEFAULT '[]',     -- JSON 推送通道列表
    enabled       INTEGER NOT NULL DEFAULT 1,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_fired_at TIMESTAMP,
    fire_count    INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_rules_symbol ON rules(symbol);
CREATE INDEX IF NOT EXISTS idx_rules_enabled ON rules(enabled);

CREATE TABLE IF NOT EXISTS signals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol       TEXT    NOT NULL,
    signal_type  TEXT    NOT NULL,                   -- BUY/SELL/HOLD 或策略名
    strength     REAL,                               -- 0~1 置信度
    context      TEXT,                               -- JSON 快照（当时价格、因子值等）
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_created ON signals(created_at);

CREATE TABLE IF NOT EXISTS ml_models (
    id            TEXT    PRIMARY KEY,                  -- UUID 如 m_20260712_001
    name          TEXT,
    universe      TEXT,                                  -- csi300 / 自定义列表
    features      TEXT,                                  -- JSON 特征列表 inline（problem 17 冶议）
    algo          TEXT,                                  -- lightgbm / xgboost / linear / mlp
    horizon       INTEGER,                               -- 预测窗口（天）
    trained_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    train_period  TEXT,                                  -- '2020-01-01~2025-12-31'
    valid_period  TEXT,
    metrics       TEXT,                                  -- JSON 评估指标
    model_path    TEXT,                                  -- 外部模型文件路径
    is_active     INTEGER NOT NULL DEFAULT 0,           -- 按 universe 粒度激活（problem 17）
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS ml_predictions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id     TEXT    NOT NULL,
    symbol       TEXT    NOT NULL,
    date         DATE    NOT NULL,
    score        REAL,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_id) REFERENCES ml_models(id)
);

CREATE INDEX IF NOT EXISTS idx_ml_pred_lookup ON ml_predictions(model_id, symbol, date);

CREATE TABLE IF NOT EXISTS ml_runs (
    id           TEXT    PRIMARY KEY,                   -- UUID
    model_id     TEXT,
    started_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    finished_at  TIMESTAMP,
    status       TEXT,                                  -- running / success / failed
    error        TEXT,
    config       TEXT                                   -- JSON 训练配置
);

CREATE TABLE IF NOT EXISTS scheduled_jobs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    UNIQUE NOT NULL,              -- 用户起名，唯一
    cron_expr     TEXT    NOT NULL,                     -- 标准 cron 表达式（分 时 日 月 周）
    action        TEXT    NOT NULL,                     -- monitor_run / scan_report / custom
    params        TEXT,                                 -- JSON 参数（如 scan 的 market/sort_by/top_n）
    channels      TEXT    NOT NULL DEFAULT '[]',         -- JSON 推送通道列表
    enabled       INTEGER NOT NULL DEFAULT 1,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_run_at   TIMESTAMP,
    last_run_status TEXT,                               -- success / failed / timeout
    last_run_error TEXT,
    run_count     INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_scheduled_enabled ON scheduled_jobs(enabled);
"""

_SCHEMA_CACHE = """
CREATE TABLE IF NOT EXISTS bar_cache (
    symbol      TEXT    NOT NULL,
    date        DATE    NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    volume      REAL,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, date)
);
"""


def _home_dir() -> Path:
    """Return the EternityQuant home directory, creating it if missing."""
    DEFAULT_HOME.mkdir(parents=True, exist_ok=True)
    (DEFAULT_HOME / "logs").mkdir(exist_ok=True)
    return DEFAULT_HOME


def _connect(name: str) -> sqlite3.Connection:
    """Open a connection to one of the two SQLite files and ensure schema."""
    path = _home_dir() / name
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    schema = _SCHEMA_STATE if name == "eternityquant.db" else _SCHEMA_CACHE
    conn.executescript(schema)
    conn.commit()
    return conn


def get_state_conn() -> sqlite3.Connection:
    """Connection to the state database (watchlist/portfolio/rules/...)."""
    return _connect("eternityquant.db")


def get_cache_conn() -> sqlite3.Connection:
    """Connection to the market cache database (bar_cache)."""
    return _connect("market_cache.db")


def execute(query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    """Run a SELECT on the state DB and return rows."""
    with get_state_conn() as conn:
        return conn.execute(query, params).fetchall()


def execute_write(query: str, params: tuple[Any, ...] = ()) -> int:
    """Run an INSERT/UPDATE/DELETE on the state DB, return lastrowid."""
    with get_state_conn() as conn:
        cur = conn.execute(query, params)
        conn.commit()
        return cur.lastrowid
