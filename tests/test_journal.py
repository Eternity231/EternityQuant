"""纸面交易日志 + 每日晨报（v0.32）。纯逻辑，注入假行情，无网络。"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from eq.core import briefing, journal
from eq.strategy import BUY, HOLD, SELL


def _bars(closes, start="2026-01-05"):
    idx = pd.bdate_range(start, periods=len(closes))
    c = np.asarray(closes, dtype="float64")
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99,
                         "close": c, "volume": np.full(len(c), 1e6)}, index=idx)


def _fetch_factory(data: dict[str, pd.DataFrame]):
    def fetch(sym, days):
        if sym not in data:
            raise KeyError(sym)
        return data[sym]
    return fetch


# ====================== 记录 ======================

def test_record_and_dedup(tmp_db):
    recos = [{"symbol": "600519.SH", "date": dt.date(2026, 1, 5), "price": 100.0}]
    assert journal.record(recos, "s1") == 1
    assert journal.record(recos, "s1") == 0, "同日同策略同标的应去重"
    assert journal.record(recos, "s2") == 1, "不同策略是独立记录"


def test_record_skips_invalid_price(tmp_db):
    recos = [{"symbol": "A", "date": dt.date(2026, 1, 5), "price": 0.0},
             {"symbol": "B", "date": dt.date(2026, 1, 5), "price": 10.0}]
    assert journal.record(recos, "s") == 1


def test_record_rejects_bad_horizon(tmp_db):
    with pytest.raises(ValueError, match="必须为正"):
        journal.record([], "s", horizon_days=0)


def test_record_accepts_timestamp_dates(tmp_db):
    assert journal.record(
        [{"symbol": "A", "date": pd.Timestamp("2026-01-05"), "price": 10.0}], "s") == 1


# ====================== 结算 ======================

def test_evaluate_exits_at_exact_horizon_bar(tmp_db):
    """结算价必须是推荐日之后**第 horizon 个交易日**的收盘价。"""
    closes = [100, 101, 102, 103, 104, 105, 106]
    data = {"A": _bars(closes), "BENCH": _bars([50, 50, 51, 52, 53, 54, 55])}
    reco_date = data["A"].index[1].date()      # 收盘 101 的那天
    journal.record([{"symbol": "A", "date": reco_date, "price": 101.0}],
                   "s", horizon_days=3, benchmark="BENCH", benchmark_price=50.0)
    settled = journal.evaluate_due(_fetch_factory(data))
    assert len(settled) == 1
    t = settled[0]
    # reco 之后第 3 个交易日收盘 = closes[4] = 104
    assert t["ret"] == pytest.approx(104 / 101 - 1)
    assert t["bench_ret"] == pytest.approx(53 / 50 - 1)
    assert t["excess"] == pytest.approx((104 / 101) - (53 / 50))


def test_evaluate_not_due_stays_open(tmp_db):
    data = {"A": _bars([100, 101, 102])}
    journal.record([{"symbol": "A", "date": data["A"].index[0].date(), "price": 100.0}],
                   "s", horizon_days=10)
    assert journal.evaluate_due(_fetch_factory(data)) == []
    assert journal.scoreboard("s")["n_open"] == 1


def test_evaluate_missing_symbol_stays_open(tmp_db):
    journal.record([{"symbol": "GONE", "date": dt.date(2026, 1, 5), "price": 10.0}],
                   "s", horizon_days=1)
    assert journal.evaluate_due(_fetch_factory({})) == []
    assert journal.scoreboard("s")["n_open"] == 1


def test_evaluate_without_benchmark_price(tmp_db):
    """记录时拿不到基准价 → 结算只算绝对收益，excess 为 None。"""
    data = {"A": _bars([100, 102, 104, 106])}
    journal.record([{"symbol": "A", "date": data["A"].index[0].date(), "price": 100.0}],
                   "s", horizon_days=2, benchmark_price=None)
    settled = journal.evaluate_due(_fetch_factory(data))
    assert len(settled) == 1
    assert settled[0]["excess"] is None


def test_evaluate_is_idempotent(tmp_db):
    data = {"A": _bars([100, 101, 102, 103])}
    journal.record([{"symbol": "A", "date": data["A"].index[0].date(), "price": 100.0}],
                   "s", horizon_days=2, benchmark_price=None)
    assert len(journal.evaluate_due(_fetch_factory(data))) == 1
    assert journal.evaluate_due(_fetch_factory(data)) == [], "已结算的不该再结算"


# ====================== 战绩牌 ======================

def _seed_closed(tmp_db, rets_and_bench):
    """直接造一批已结算记录。rets_and_bench = [(ret, bench_ret), ...]"""
    from eq.db import get_state_conn

    with get_state_conn() as conn:
        for i, (r, b) in enumerate(rets_and_bench):
            conn.execute(
                """INSERT INTO paper_recos
                   (reco_date, symbol, strategy, entry_price, horizon_days,
                    benchmark, benchmark_entry, status, exit_date, exit_price, benchmark_exit)
                   VALUES (?, ?, 'test', 100.0, 5, 'B', 100.0, 'closed', ?, ?, ?)""",
                (f"2026-01-{i + 1:02d}", f"S{i:02d}", f"2026-02-{i + 1:02d}",
                 100.0 * (1 + r), 100.0 * (1 + b)),
            )
        conn.commit()


def test_scoreboard_math(tmp_db):
    _seed_closed(tmp_db, [(0.05, 0.02), (0.01, 0.02), (-0.02, -0.01), (0.04, 0.01)])
    sb = journal.scoreboard("test")
    assert sb["n_closed"] == 4
    assert sb["win_rate"] == pytest.approx(0.75)
    assert sb["ret_mean"] == pytest.approx((0.05 + 0.01 - 0.02 + 0.04) / 4)
    ex = np.array([0.03, -0.01, -0.01, 0.03])
    assert sb["excess_mean"] == pytest.approx(ex.mean())
    assert sb["beat_bench_rate"] == pytest.approx(0.5)
    t_expected = ex.mean() / ex.std(ddof=1) * np.sqrt(4)
    assert sb["excess_t"] == pytest.approx(t_expected)


def test_scoreboard_empty(tmp_db):
    sb = journal.scoreboard()
    assert sb["n_closed"] == 0
    assert "还没有" in journal.format_scoreboard(sb)


def test_scoreboard_filters_by_strategy(tmp_db):
    _seed_closed(tmp_db, [(0.05, 0.02)])
    assert journal.scoreboard("test")["n_closed"] == 1
    assert journal.scoreboard("别的策略")["n_closed"] == 0


def test_scoreboard_estimates_n_for_significance(tmp_db):
    _seed_closed(tmp_db, [(0.03, 0.01), (0.01, 0.02), (0.04, 0.01), (0.02, 0.015)])
    sb = journal.scoreboard("test")
    if sb["excess_mean"] > 0 and abs(sb["excess_t"]) < 2:
        assert sb["n_for_significance"] and sb["n_for_significance"] > sb["n_vs_bench"]


def test_format_scoreboard_verdicts(tmp_db):
    # 显著跑赢：超额稳定为正
    _seed_closed(tmp_db, [(0.05, 0.01)] * 12)
    out = journal.format_scoreboard(journal.scoreboard("test"))
    assert "显著" in out and "跑赢" in out


def test_recent_closed(tmp_db):
    _seed_closed(tmp_db, [(0.05, 0.02), (-0.01, 0.0)])
    rows = journal.recent_closed(limit=10, strategy="test")
    assert len(rows) == 2
    assert all("ret" in r and "excess" in r for r in rows)


# ====================== 晨报：信号变化 ======================

def _sig_fn(pattern):
    """按预置三态列表出信号。"""
    def fn(df):
        s = pd.Series(HOLD, index=df.index, dtype=object)
        for i, v in enumerate(pattern[-len(df):]):
            s.iloc[i] = v
        return s
    return fn


def test_detect_enter_on_last_bar():
    bars = {"A": _bars([100] * 40)}
    pattern = [HOLD] * 39 + [BUY]
    assert briefing.detect_signal_changes(bars, _sig_fn(pattern))["A"] == "enter"


def test_detect_exit_on_last_bar():
    bars = {"A": _bars([100] * 40)}
    pattern = [BUY] + [HOLD] * 38 + [SELL]
    assert briefing.detect_signal_changes(bars, _sig_fn(pattern))["A"] == "exit"


def test_detect_holding_not_reported_as_enter():
    """早已持有、今天没翻转 → holding，不该被当成新买入。"""
    bars = {"A": _bars([100] * 40)}
    pattern = [BUY] + [HOLD] * 39
    assert briefing.detect_signal_changes(bars, _sig_fn(pattern))["A"] == "holding"


def test_detect_flat():
    bars = {"A": _bars([100] * 40)}
    assert briefing.detect_signal_changes(bars, _sig_fn([HOLD] * 40))["A"] == "flat"


def test_detect_numeric_positions():
    bars = {"A": _bars([100] * 40)}
    def fn(df):
        s = pd.Series(0.0, index=df.index)
        s.iloc[-1] = 0.6
        return s
    assert briefing.detect_signal_changes(bars, fn)["A"] == "enter"


def test_detect_skips_short_and_broken():
    bars = {"short": _bars([100] * 5), "boom": _bars([100] * 40)}
    def fn(df):
        if len(df) == 40:
            raise RuntimeError("炸")
        return pd.Series(HOLD, index=df.index)
    assert briefing.detect_signal_changes(bars, fn) == {}


# ====================== 晨报：止损 / 大盘 / 推荐打包 ======================

def test_stop_breaches_classification():
    positions = [
        {"symbol": "破位", "stop_loss": 10.0, "current_price": 9.5},
        {"symbol": "逼近", "stop_loss": 10.0, "current_price": 10.15},
        {"symbol": "安全", "stop_loss": 10.0, "current_price": 12.0},
        {"symbol": "裸奔", "stop_loss": None, "current_price": 8.0},
    ]
    out = briefing.stop_breaches(positions)
    assert [p["symbol"] for p in out["breached"]] == ["破位"]
    assert [p["symbol"] for p in out["near"]] == ["逼近"]
    assert [p["symbol"] for p in out["no_stop"]] == ["裸奔"]


def test_market_status_gate():
    up = _bars(list(np.linspace(100, 200, 300)))
    ms = briefing.market_status(up, ma_period=60)
    assert ms["gate_open"] is True
    assert ms["dist_ma_pct"] > 0
    down = _bars(list(np.linspace(200, 100, 300)))
    assert briefing.market_status(down, ma_period=60)["gate_open"] is False


def test_market_status_none_input():
    assert briefing.market_status(None) is None
    assert briefing.market_status(_bars([100] * 5)) is None


def test_build_recos_only_enters():
    bars = {"A": _bars([100, 110]), "B": _bars([50, 55])}
    changes = {"A": "enter", "B": "holding"}
    bench = _bars([3000, 3050])
    recos, bench_px = briefing.build_recos(bars, changes, bench)
    assert len(recos) == 1
    assert recos[0]["symbol"] == "A"
    assert recos[0]["price"] == pytest.approx(110.0)
    assert recos[0]["date"] == bars["A"].index[-1].date()
    assert bench_px == pytest.approx(3050.0)


def test_build_recos_without_benchmark():
    bars = {"A": _bars([100, 110])}
    recos, bench_px = briefing.build_recos(bars, {"A": "enter"}, None)
    assert len(recos) == 1 and bench_px is None


# ====================== 端到端：记录 → 结算 → 战绩 ======================

def test_full_paper_loop(tmp_db):
    """模拟三天晨报的完整闭环。"""
    closes = list(np.linspace(100, 120, 15))
    data = {"600519.SH": _bars(closes), "000300.SH": _bars(list(np.linspace(3000, 3100, 15)))}

    # 第 3 天：策略发买入信号 → 记录
    d3 = data["600519.SH"].index[2].date()
    journal.record([{"symbol": "600519.SH", "date": d3, "price": float(closes[2])}],
                   "trend_vote", horizon_days=5, benchmark="000300.SH",
                   benchmark_price=float(data["000300.SH"]["close"].iloc[2]))

    # 次日跑晨报：只有 4 根后续 bar 可见 → 还没到期
    short = {k: v.iloc[:7] for k, v in data.items()}     # reco 后仅 4 根
    assert journal.evaluate_due(_fetch_factory(short)) == []
    assert journal.scoreboard("trend_vote")["n_open"] == 1

    # 一周后再跑：后续 bar ≥ horizon → 结算，出场价 = reco 后第 5 根收盘
    settled = journal.evaluate_due(_fetch_factory(data))
    assert len(settled) == 1
    t = settled[0]
    assert t["ret"] == pytest.approx(closes[2 + 5] / closes[2] - 1)
    assert t["excess"] is not None

    sb = journal.scoreboard("trend_vote")
    assert sb["n_closed"] == 1 and sb["n_open"] == 0
    assert sb["n_vs_bench"] == 1
    assert isinstance(journal.format_scoreboard(sb), str)
