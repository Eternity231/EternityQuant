"""数据导出（v0.24 新增）。

把库里的自选 / 持仓 / 交易流水 / 监控规则 / 信号 / 回测记录导出成 CSV 或
Excel，方便拿去做税务申报、对账、或者丢进 Excel 自己画图。

CSV 默认用 ``utf-8-sig`` 编码——Excel 直接双击打开 UTF-8 CSV 会把中文显示成
乱码，加 BOM 才认。
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from eq.core import monitor as mon_svc
from eq.core import portfolio as pf_svc
from eq.core import watchlist as wl_svc
from eq.db import execute

CSV_ENCODING = "utf-8-sig"  # 带 BOM，Excel 双击不乱码


def _watchlist() -> pd.DataFrame:
    return pd.DataFrame(wl_svc.list_all())


def _positions() -> pd.DataFrame:
    """当前持仓（带实时行情与浮盈，拉不到行情时退化为成本价）。"""
    try:
        return pd.DataFrame(pf_svc.summary()["positions"])
    except Exception:
        return pd.DataFrame(pf_svc.list_open())


def _positions_closed() -> pd.DataFrame:
    return pd.DataFrame(pf_svc.list_closed(limit=100000))


def _trades() -> pd.DataFrame:
    rows = execute(
        "SELECT id, symbol, action, shares, price, executed_at, note "
        "FROM trade_history ORDER BY executed_at DESC, id DESC"
    )
    return pd.DataFrame([dict(r) for r in rows])


def _rules() -> pd.DataFrame:
    rules = mon_svc.list_rules()
    for r in rules:
        r["params"] = str(r["params"])
        r["channels"] = ",".join(r["channels"])
    return pd.DataFrame(rules)


def _signals() -> pd.DataFrame:
    sigs = mon_svc.recent_signals(limit=100000)
    for s in sigs:
        ctx = s.get("context") or {}
        s["title"] = ctx.get("title", "")
        s["body"] = str(ctx.get("body", "")).replace("\n", " / ")
        s.pop("context", None)
    return pd.DataFrame(sigs)


def _backtests() -> pd.DataFrame:
    from eq.backtest.store import list_runs

    runs = list_runs(limit=100000)
    flat = []
    for r in runs:
        row = {k: v for k, v in r.items() if k != "metrics"}
        row.update({f"metric_{k}": v for k, v in (r.get("metrics") or {}).items()})
        flat.append(row)
    return pd.DataFrame(flat)


DATASETS: dict[str, tuple[Callable[[], pd.DataFrame], str]] = {
    "watchlist": (_watchlist, "自选股"),
    "positions": (_positions, "当前持仓"),
    "closed": (_positions_closed, "已清仓"),
    "trades": (_trades, "交易流水"),
    "rules": (_rules, "监控规则"),
    "signals": (_signals, "触发信号"),
    "backtests": (_backtests, "回测记录"),
}


def export(
    datasets: list[str] | None = None,
    out_dir: str | Path | None = None,
    fmt: str = "csv",
) -> dict[str, Any]:
    """导出指定数据集。

    Args:
        datasets: :data:`DATASETS` 里的名字；``None`` = 全部
        out_dir: 输出目录，缺省 ``.eternityquant/exports/<日期>/``
        fmt: ``csv``（一表一文件）或 ``excel``（一个 xlsx 多 sheet）

    Returns:
        ``{"out_dir": str, "files": [...], "rows": {name: n}, "skipped": [...]}``
    """
    from eq.db import DEFAULT_HOME

    names = list(datasets) if datasets else list(DATASETS)
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        raise ValueError(f"未知数据集 {unknown}，可选：{sorted(DATASETS)}")
    if fmt not in ("csv", "excel"):
        raise ValueError(f"fmt 只能是 csv / excel，收到 {fmt}")

    if out_dir is None:
        out_dir = DEFAULT_HOME / "exports" / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    rows: dict[str, int] = {}
    skipped: list[str] = []
    for name in names:
        loader = DATASETS[name][0]
        try:
            df = loader()
        except Exception as e:
            skipped.append(f"{name}（{type(e).__name__}: {e}）")
            continue
        if df is None or df.empty:
            skipped.append(f"{name}（无数据）")
            continue
        frames[name] = df
        rows[name] = len(df)

    files: list[str] = []
    if fmt == "csv":
        for name, df in frames.items():
            path = out / f"{name}.csv"
            df.to_csv(path, index=False, encoding=CSV_ENCODING)
            files.append(str(path))
    else:
        path = out / "eternityquant.xlsx"
        try:
            with pd.ExcelWriter(path) as writer:
                for name, df in frames.items():
                    # Excel sheet 名上限 31 字符
                    df.to_excel(writer, sheet_name=name[:31], index=False)
            files.append(str(path))
        except ImportError as e:
            raise RuntimeError(
                "导出 Excel 需要 openpyxl：pip install openpyxl（或改用 --format csv）"
            ) from e

    return {"out_dir": str(out), "files": files, "rows": rows, "skipped": skipped}
