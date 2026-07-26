"""行情本地缓存（``market_cache.db`` 的 ``bar_cache`` 表）。

建库时就定义了 ``bar_cache`` 表，但此前没有任何代码读写它——每次
``eq watch`` / ``eq monitor run`` / ``eq backtest`` 都在打网络重拉同一段日线。
本模块把它接上：

- :func:`load_bars`   从缓存读某标的某区间的日线
- :func:`save_bars`   把拉到的日线 upsert 进缓存
- :func:`is_fresh`    判断某标的缓存是否在 TTL 内（免网络）
- :func:`stats` / :func:`clear` 供 ``eq cache`` 命令用

缓存是纯加速层：任何一步失败都吞掉异常走网络，绝不因为缓存挂了阻断主流程。
"""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from eq.db import get_cache_conn

logger = logging.getLogger(__name__)

_COLS = ["open", "high", "low", "close", "volume"]

# 默认 TTL：盘中数据 6 小时后视为过期。日线在收盘后才定型，
# 6 小时能覆盖「同一交易日内反复查」而不至于拿到隔夜的陈旧数据。
DEFAULT_TTL_SECONDS = 6 * 3600


def load_bars(symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """从缓存读 ``[start, end]`` 的日线。无数据返回空 DataFrame。"""
    try:
        with get_cache_conn() as conn:
            rows = conn.execute(
                "SELECT date, open, high, low, close, volume FROM bar_cache "
                "WHERE symbol = ? AND date >= ? AND date <= ? ORDER BY date",
                (symbol, start.isoformat(), end.isoformat()),
            ).fetchall()
    except Exception as e:  # pragma: no cover - 缓存永不阻断主流程
        logger.debug("缓存读取失败 %s：%s", symbol, e)
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([tuple(r) for r in rows], columns=["date", *_COLS])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def save_bars(symbol: str, df: pd.DataFrame) -> int:
    """把日线 upsert 进缓存，返回写入行数。df 需以日期为索引。"""
    if df is None or df.empty:
        return 0
    missing = [c for c in _COLS if c not in df.columns]
    if missing:
        return 0
    try:
        idx = pd.to_datetime(df.index)
    except Exception:
        return 0
    payload = []
    for ts, vals in zip(idx, df[_COLS].to_numpy(dtype="float64", na_value=float("nan")), strict=False):
        o, h, low_, c, v = vals
        if pd.isna(c):
            continue
        payload.append((symbol, ts.date().isoformat(), _f(o), _f(h), _f(low_), _f(c), _f(v)))
    if not payload:
        return 0
    try:
        with get_cache_conn() as conn:
            conn.executemany(
                "INSERT INTO bar_cache (symbol, date, open, high, low, close, volume, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP) "
                "ON CONFLICT(symbol, date) DO UPDATE SET "
                "open=excluded.open, high=excluded.high, low=excluded.low, "
                "close=excluded.close, volume=excluded.volume, updated_at=CURRENT_TIMESTAMP",
                payload,
            )
            conn.execute(
                "INSERT INTO cache_meta (symbol, fetched_at, first_date, last_date, rows) "
                "VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?) "
                "ON CONFLICT(symbol) DO UPDATE SET "
                "fetched_at=CURRENT_TIMESTAMP, "
                "first_date=MIN(cache_meta.first_date, excluded.first_date), "
                "last_date=MAX(cache_meta.last_date, excluded.last_date), "
                "rows=excluded.rows",
                (symbol, payload[0][1], payload[-1][1], len(payload)),
            )
            conn.commit()
    except Exception as e:  # pragma: no cover
        logger.debug("缓存写入失败 %s：%s", symbol, e)
        return 0
    return len(payload)


def _f(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


def is_fresh(symbol: str, start: dt.date, end: dt.date, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> bool:
    """缓存是否「新鲜到可以直接用」。

    条件：上次真实拉取在 TTL 内，且缓存覆盖到了 ``end``（或 ``end`` 之后最近的
    一个已缓存交易日——周末/节假日 ``end`` 本来就没有数据，用 4 天容忍窗）。
    """
    try:
        with get_cache_conn() as conn:
            row = conn.execute(
                "SELECT fetched_at, first_date, last_date FROM cache_meta WHERE symbol = ?",
                (symbol,),
            ).fetchone()
    except Exception:  # pragma: no cover
        return False
    if row is None or not row["fetched_at"]:
        return False
    fetched = _parse_ts(row["fetched_at"])
    if fetched is None:
        return False
    age = (dt.datetime.now(dt.timezone.utc) - fetched).total_seconds()
    if age > ttl_seconds or age < -60:
        return False
    first = _parse_date(row["first_date"])
    last = _parse_date(row["last_date"])
    if first is None or last is None:
        return False
    # 起点必须被覆盖；终点允许落后 4 天（周末 + 一天假期）
    return first <= start and (end - last).days <= 4


def _parse_ts(v) -> dt.datetime | None:
    if isinstance(v, dt.datetime):
        d = v
    else:
        try:
            d = dt.datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        except ValueError:
            return None
    # SQLite CURRENT_TIMESTAMP 是 UTC 且不带 tzinfo
    return d.replace(tzinfo=dt.timezone.utc) if d.tzinfo is None else d


def _parse_date(v) -> dt.date | None:
    if isinstance(v, dt.datetime):
        return v.date()
    if isinstance(v, dt.date):
        return v
    try:
        return dt.date.fromisoformat(str(v)[:10])
    except ValueError:
        return None


def stats() -> dict:
    """缓存概况：标的数 / 总行数 / 日期跨度 / 文件大小。"""
    from eq.db import DEFAULT_HOME

    out = {"symbols": 0, "rows": 0, "first_date": None, "last_date": None, "size_mb": 0.0, "per_symbol": []}
    try:
        with get_cache_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT symbol) s, COUNT(*) n, MIN(date) f, MAX(date) l FROM bar_cache"
            ).fetchone()
            out.update(symbols=row["s"] or 0, rows=row["n"] or 0,
                       first_date=row["f"], last_date=row["l"])
            out["per_symbol"] = [
                {"symbol": r["symbol"], "rows": r["n"], "first": r["f"], "last": r["l"]}
                for r in conn.execute(
                    "SELECT symbol, COUNT(*) n, MIN(date) f, MAX(date) l "
                    "FROM bar_cache GROUP BY symbol ORDER BY symbol"
                ).fetchall()
            ]
    except Exception as e:  # pragma: no cover
        logger.debug("缓存统计失败：%s", e)
    db_file = DEFAULT_HOME / "market_cache.db"
    if db_file.exists():
        out["size_mb"] = round(db_file.stat().st_size / 1024 / 1024, 2)
    return out


def clear(symbol: str | None = None) -> int:
    """清缓存。``symbol=None`` 清全部。返回删除行数。"""
    try:
        with get_cache_conn() as conn:
            if symbol:
                cur = conn.execute("DELETE FROM bar_cache WHERE symbol = ?", (symbol,))
                conn.execute("DELETE FROM cache_meta WHERE symbol = ?", (symbol,))
            else:
                cur = conn.execute("DELETE FROM bar_cache")
                conn.execute("DELETE FROM cache_meta")
            conn.commit()
            return cur.rowcount
    except Exception as e:  # pragma: no cover
        logger.debug("缓存清理失败：%s", e)
        return 0
