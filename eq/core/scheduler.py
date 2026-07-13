"""定时推送服务（APScheduler）。

进程常驻调度器，盘后定时跑 monitor run / 每日扫描报告等。
任务配置持久化到 scheduled_jobs 表，进程退出不丢，重启时恢复。

cron_expr 用标准 5 字段格式：分 时 日 月 周
  - 0 15 * * 1-5     工作日 15:00（A 股收盘后）
  - 0 16 * * 1-5     工作日 16:00（盘后整理完毕）
  - 30 9 * * 1-5     工作日 9:30（开盘前速览）
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from eq.core import monitor as mon_svc
from eq.core.notifier import dispatch
from eq.db import execute, execute_write, get_state_conn

logger = logging.getLogger(__name__)

# ---------- action 处理器 ----------

_ACTIONS = {}


def register_action(name: str, func):
    """注册一个 action 处理器：func(params: dict, channels: list[str]) -> (title, body)。"""
    _ACTIONS[name] = func


def _action_monitor_run(params: dict, channels: list[str]) -> tuple[str, str]:
    """跑所有 enabled 监控规则，触发则推送。"""
    fired = mon_svc.run_all()
    title = "EternityQuant 监控扫描"
    body = f"扫描完成，触发 {fired} 条规则" + ("\n（无触发）" if fired == 0 else "")
    return title, body


def _action_scan_report(params: dict, channels: list[str]) -> tuple[str, str]:
    """生成每日扫描报告并推送。"""
    from eq.core.scanner import scan_a_share
    market = params.get("market", "A")
    sort_by = params.get("sort_by", "change_pct")
    top_n = params.get("top_n", 30)
    if market != "A":
        return "扫描报告", f"市场 {market} 待集成"
    df = scan_a_share(sort_by=sort_by, top_n=top_n)
    title = "EternityQuant 每日扫描报告"
    body = f"按 {sort_by} 排序，前 {len(df)} 名：\n"
    for _, row in df.head(10).iterrows():
        arrow = "▲" if row["change_pct"] >= 0 else "▼"
        body += f"{row['symbol']} {row['name'][:6]} {row['close']:.2f} {arrow}{row['change_pct']:+.2f}%\n"
    return title, body


register_action("monitor_run", _action_monitor_run)
register_action("scan_report", _action_scan_report)


# ---------- CRUD ----------

def add_job(
    name: str,
    cron_expr: str,
    action: str,
    params: dict[str, Any] | None = None,
    channels: list[str] | None = None,
) -> int:
    """注册定时任务。返回 job_id。"""
    if action not in _ACTIONS:
        raise ValueError(f"未知 action {action}，可选：{sorted(_ACTIONS)}")
    # 校验 cron 表达式
    try:
        CronTrigger.from_crontab(cron_expr)
    except ValueError as e:
        raise ValueError(f"cron 表达式无效：{e}") from e
    if channels is None:
        channels = ["desktop"]
    rowid = execute_write(
        """INSERT INTO scheduled_jobs (name, cron_expr, action, params, channels)
           VALUES (?, ?, ?, ?, ?)""",
        (name, cron_expr, action, json.dumps(params or {}, ensure_ascii=False), json.dumps(channels)),
    )
    return rowid


def remove_job(job_id: int) -> bool:
    with get_state_conn() as conn:
        cur = conn.execute("DELETE FROM scheduled_jobs WHERE id = ?", (job_id,))
        conn.commit()
        return cur.rowcount > 0


def set_enabled(job_id: int, enabled: bool) -> bool:
    with get_state_conn() as conn:
        cur = conn.execute("UPDATE scheduled_jobs SET enabled = ? WHERE id = ?", (1 if enabled else 0, job_id))
        conn.commit()
        return cur.rowcount > 0


def list_jobs(enabled_only: bool = False) -> list[dict[str, Any]]:
    q = "SELECT id, name, cron_expr, action, params, channels, enabled, created_at, last_run_at, last_run_status, run_count FROM scheduled_jobs"
    if enabled_only:
        q += " WHERE enabled = 1"
    q += " ORDER BY id"
    rows = execute(q)
    out = []
    for r in rows:
        d = {k: r[k] for k in r.keys()}
        d["params"] = json.loads(d["params"] or "{}")
        d["channels"] = json.loads(d["channels"] or "[]")
        out.append(d)
    return out


def _mark_run(job_id: int, status: str, error: str = "") -> None:
    execute_write(
        """UPDATE scheduled_jobs SET last_run_at = CURRENT_TIMESTAMP,
           last_run_status = ?, last_run_error = ?, run_count = run_count + 1 WHERE id = ?""",
        (status, error or None, job_id),
    )


# ---------- 调度器 ----------

class Scheduler:
    """进程常驻调度器。daemon=True，进程退出自动清理。"""

    def __init__(self) -> None:
        self._scheduler: BackgroundScheduler | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._scheduler is not None:
                return
            sched = BackgroundScheduler(daemon=True)
            sched.start()
            self._scheduler = sched
            # 恢复所有 enabled 任务
            for job in list_jobs(enabled_only=True):
                self._add_aps_job(job)
            logger.info("调度器已启动，恢复 %d 个任务", len(list_jobs(enabled_only=True)))

    def stop(self) -> None:
        with self._lock:
            if self._scheduler is not None:
                self._scheduler.shutdown(wait=False)
                self._scheduler = None

    def _add_aps_job(self, job: dict[str, Any]) -> None:
        assert self._scheduler is not None
        trigger = CronTrigger.from_crontab(job["cron_expr"])
        self._scheduler.add_job(
            self._run_job,
            trigger=trigger,
            args=[job["id"]],
            id=f"job_{job['id']}",
            replace_existing=True,
        )

    def _remove_aps_job(self, job_id: int) -> None:
        if self._scheduler is not None:
            try:
                self._scheduler.remove_job(f"job_{job_id}")
            except KeyError:
                pass

    def reload(self) -> None:
        """重新加载所有任务（配置变更后调）。"""
        with self._lock:
            if self._scheduler is None:
                return
            # 移除所有现有任务
            existing = {j.id for j in self._scheduler.get_jobs()}
            for jid in existing:
                self._scheduler.remove_job(jid)
            # 重新添加 enabled 任务
            for job in list_jobs(enabled_only=True):
                self._add_aps_job(job)

    def _run_job(self, job_id: int) -> None:
        """任务触发时的执行逻辑。"""
        rows = execute(
            "SELECT id, name, action, params, channels FROM scheduled_jobs WHERE id = ? AND enabled = 1",
            (job_id,),
        )
        if not rows:
            return
        row = rows[0]
        action_name = row["action"]
        params = json.loads(row["params"] or "{}")
        channels = json.loads(row["channels"] or "[]")
        handler = _ACTIONS.get(action_name)
        if handler is None:
            _mark_run(job_id, "failed", f"未知 action {action_name}")
            return
        try:
            title, body = handler(params, channels)
            if channels:
                dispatch(channels, title, body, job_name=row["name"])
            _mark_run(job_id, "success")
        except Exception as e:
            logger.exception("任务 %s 执行失败", row["name"])
            _mark_run(job_id, "failed", str(e))

    def run_now(self, job_id: int) -> bool:
        """立即执行某任务一次（不等触发）。返回是否找到并执行。"""
        rows = execute("SELECT id FROM scheduled_jobs WHERE id = ?", (job_id,))
        if not rows:
            return False
        self._run_job(job_id)
        return True


# 全局单例（CLI daemon 命令持有）
_scheduler_instance: Scheduler | None = None


def get_scheduler() -> Scheduler:
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = Scheduler()
    return _scheduler_instance
