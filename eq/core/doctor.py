"""环境体检（``eq doctor``）。

一条命令回答"我这套东西现在到底能不能跑"：Python 版本、可选依赖装没装、
数据目录在哪有多大、数据库通不通、推送通道配没配、行情源连不连得上。

设计原则：**任何一项失败都不能让体检本身崩掉**，每项独立 try/except，
体检报告要在环境最烂的时候也能打出来——那正是最需要它的时候。
"""

from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path
from typing import Any

OK, WARN, FAIL = "ok", "warn", "fail"

# (模块名, 是否必需, 用途说明)
_DEPS = [
    ("pandas", True, "所有数据处理"),
    ("numpy", True, "数值计算"),
    ("typer", True, "CLI 框架"),
    ("requests", True, "东财 push2his 数据源"),
    ("akshare", False, "A股/港股/美股扫描 + fallback 行情"),
    ("yfinance", False, "港股/美股/加密行情主源"),
    ("baostock", False, "A 股日线主源"),
    ("apscheduler", False, "eq scheduler 定时任务"),
    ("streamlit", False, "eq dash 网页仪表盘"),
    ("plotly", False, "仪表盘图表"),
    ("pyarrow", False, "回测结果 parquet 外存"),
    ("lightgbm", False, "eq ml train LightGBM"),
    ("torch", False, "eq ml train 深度模型"),
    ("qlib", False, "eq ml train A 股 workflow"),
    ("ccxt", False, "eq scan CRYPTO"),
    ("openpyxl", False, "eq export --format excel"),
]


def _installed(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def _dir_size(p: Path) -> tuple[int, int]:
    """返回 (文件数, 字节数)。目录不存在返回 (0, 0)。"""
    if not p.exists():
        return 0, 0
    n = size = 0
    try:
        for f in p.rglob("*"):
            if f.is_file():
                n += 1
                size += f.stat().st_size
    except OSError:
        pass
    return n, size


def check() -> list[dict[str, Any]]:
    """跑全部体检项，返回 ``[{section, name, status, detail}]``。"""
    items: list[dict[str, Any]] = []

    def add(section: str, name: str, status: str, detail: str):
        items.append({"section": section, "name": name, "status": status, "detail": detail})

    # ---- 运行环境 ----
    v = sys.version_info
    add("环境", "Python", OK if v >= (3, 10) else FAIL,
        f"{platform.python_version()}（要求 ≥ 3.10）")
    add("环境", "平台", OK, f"{platform.system()} {platform.release()}")
    add("环境", "解释器", OK, sys.executable)

    # ---- 依赖 ----
    for mod, required, why in _DEPS:
        if _installed(mod):
            ver = ""
            try:
                from importlib.metadata import version

                ver = version(mod)
            except Exception:
                pass
            add("依赖", mod, OK, f"{ver or '已安装'} — {why}")
        else:
            add("依赖", mod, FAIL if required else WARN,
                f"未安装 — {why}" + ("" if required else "（可选）"))

    # ---- CUDA ----
    if _installed("torch"):
        try:
            import torch

            if torch.cuda.is_available():
                add("依赖", "CUDA", OK,
                    f"{torch.cuda.device_count()} 张卡 — {torch.cuda.get_device_name(0)}")
            else:
                add("依赖", "CUDA", WARN, "torch 装的是 CPU 版或无可用 GPU（训练会很慢）")
        except Exception as e:
            add("依赖", "CUDA", WARN, f"探测失败：{type(e).__name__}: {e}")

    # ---- 数据库 ----
    try:
        from eq.db import DEFAULT_HOME, execute

        add("数据", "HOME 目录", OK, str(DEFAULT_HOME))
        counts = {}
        for table in ("watchlist", "portfolio", "trade_history", "rules",
                      "signals", "ml_models", "backtest_runs", "scheduled_jobs"):
            try:
                counts[table] = execute(f"SELECT COUNT(*) c FROM {table}")[0]["c"]
            except Exception:
                counts[table] = "?"
        add("数据", "状态库", OK,
            "  ".join(f"{k}={v}" for k, v in counts.items()))
    except Exception as e:
        add("数据", "状态库", FAIL, f"{type(e).__name__}: {e}")

    # ---- 行情缓存 ----
    try:
        from eq.data.cache import stats

        s = stats()
        add("数据", "行情缓存", OK if s["rows"] else WARN,
            f"{s['symbols']} 只 / {s['rows']} 行 / {s['size_mb']} MB"
            + (f" / {s['first_date']}~{s['last_date']}" if s["rows"] else "（空，首次拉行情后自动填充）"))
    except Exception as e:
        add("数据", "行情缓存", WARN, f"{type(e).__name__}: {e}")

    # ---- 数据目录 ----
    try:
        from eq.data.paths import (
            HK_1M_DIR, HK_5M_DIR, HK_DAILY_DIR, QLIB_CN_DATA_DIR, US_DAILY_DIR,
        )

        for label, d in [
            ("A 股 qlib", QLIB_CN_DATA_DIR), ("港股日线", HK_DAILY_DIR),
            ("港股 5m", HK_5M_DIR), ("港股 1m", HK_1M_DIR), ("美股日线", US_DAILY_DIR),
        ]:
            n, size = _dir_size(Path(d))
            add("数据目录", label, OK if n else WARN,
                f"{n:>6} 文件  {size / 1024 / 1024:>8.1f} MB  {d}")
    except Exception as e:
        add("数据目录", "扫描", FAIL, f"{type(e).__name__}: {e}")

    # ---- 推送通道 ----
    try:
        from eq.core.notifier import available_channels

        chs = available_channels()
        add("推送", "可用通道", OK if chs else WARN,
            ", ".join(chs) or "无（在 .eternityquant/.env 配 WECHAT_WORK_WEBHOOK 可加企业微信）")
    except Exception as e:
        add("推送", "可用通道", FAIL, f"{type(e).__name__}: {e}")

    return items


def check_network(symbol: str = "600519.SH", timeout_note: bool = True) -> list[dict[str, Any]]:
    """连通性体检（会真的打网络，所以从 :func:`check` 里拆出来单独调）。"""
    items: list[dict[str, Any]] = []
    try:
        from eq.data.market import get_snapshot

        snap = get_snapshot(symbol, use_cache=False)
        items.append({
            "section": "连通性", "name": f"行情源（{symbol}）", "status": OK,
            "detail": f"{snap['date']}  收 {snap['close']:.2f}  {snap['change_pct']:+.2f}%",
        })
    except Exception as e:
        items.append({
            "section": "连通性", "name": f"行情源（{symbol}）", "status": FAIL,
            "detail": f"{type(e).__name__}: {str(e)[:120]}"
            + ("（大陆网络下 yfinance 常被限流，A 股走 baostock 更稳）" if timeout_note else ""),
        })
    return items


_ICON = {OK: "✓", WARN: "!", FAIL: "✗"}


def format_report(items: list[dict[str, Any]]) -> str:
    """格式化体检报告。"""
    lines = ["\n===== EternityQuant 环境体检 =====\n"]
    section = None
    for it in items:
        if it["section"] != section:
            section = it["section"]
            lines.append(f"\n--- {section} ---")
        lines.append(f"  {_ICON.get(it['status'], '?')} {it['name']:<16} {it['detail']}")
    n_fail = sum(1 for i in items if i["status"] == FAIL)
    n_warn = sum(1 for i in items if i["status"] == WARN)
    lines.append(
        f"\n合计 {len(items)} 项：{len(items) - n_fail - n_warn} 正常 / {n_warn} 警告 / {n_fail} 失败\n"
    )
    return "\n".join(lines)
