"""策略稳健性验证（v0.28）。纯逻辑，无网络。"""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
import pytest

from eq.backtest import robust as R
from eq.backtest.types import BacktestConfig
from eq.strategy import BUY, HOLD, SELL
from eq.strategy.signals import ema_cross


def _bars(n=600, seed=0, trend=0.0, vol=1.5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = np.maximum(100 + np.cumsum(rng.normal(trend, vol, n)), 5.0)
    noise = np.abs(rng.normal(0, 0.8, n))
    return pd.DataFrame({
        "open": close + rng.normal(0, 0.3, n), "high": close + noise,
        "low": close - noise, "close": close,
        "volume": rng.integers(1e6, 9e6, n).astype(float),
    }, index=pd.bdate_range("2023-01-01", periods=n))


@pytest.fixture
def bars():
    return _bars()


# ====================== run_once / 汇总 ======================

def test_run_once_returns_metrics(bars):
    m = R.run_once(bars, ema_cross)
    assert {"total_return", "sharpe", "max_drawdown", "num_trades"} <= set(m)


def test_run_once_swallows_failures(bars):
    def boom(_df):
        raise RuntimeError("策略炸了")
    m = R.run_once(bars, boom)
    assert m["total_return"] == 0.0, "批量评估里单个失败不该中断整轮"


def test_summarize_uses_median_not_just_mean():
    rows = [{"sharpe": 0.1, "total_return": 0.01, "max_drawdown": -0.1, "num_trades": 1}] * 9
    rows.append({"sharpe": 50.0, "total_return": 5.0, "max_drawdown": -0.5, "num_trades": 1})
    s = R._summarize(rows)
    assert s["sharpe_median"] == pytest.approx(0.1)
    assert s["sharpe_mean"] > 5, "均值被极端值带偏——这正是要看中位数的理由"
    assert s["n"] == 10


def test_summarize_empty():
    assert R._summarize([])["n"] == 0


# ====================== 多标的 ======================

def test_multi_symbol_aggregates_distribution():
    syms = {f"S{i:02d}": _bars(seed=i, trend=0.05) for i in range(10)}
    rep = R.multi_symbol(ema_cross, syms)
    assert len(rep["per_symbol"]) == 10
    assert rep["summary"]["n"] == 10
    assert 0.0 <= rep["summary"]["pct_profitable"] <= 1.0
    assert all("symbol" in r for r in rep["per_symbol"])


def test_multi_symbol_skips_too_short_series():
    syms = {"good": _bars(n=200), "tiny": _bars(n=10)}
    rep = R.multi_symbol(ema_cross, syms)
    assert [r["symbol"] for r in rep["per_symbol"]] == ["good"]


def test_multi_symbol_handles_none():
    rep = R.multi_symbol(ema_cross, {"a": None, "b": _bars(n=200)})
    assert rep["summary"]["n"] == 1


def test_format_multi_symbol_runs():
    syms = {f"S{i}": _bars(seed=i) for i in range(4)}
    out = R.format_multi_symbol(R.multi_symbol(ema_cross, syms))
    assert "多标的稳健性" in out and "盈利标的占比" in out


# ====================== Walk-Forward ======================

def test_walk_forward_windows_are_ordered_and_purged():
    df = _bars(n=1000)
    wins = R.walk_forward_windows(df, n_splits=4, train_bars=250,
                                  test_bars=60, embargo_bars=5)
    assert len(wins) == 4
    prev_start = -1
    for w in wins:
        assert w.train.stop + 5 == w.test.start, "训练段与测试段之间必须留 embargo"
        assert w.test.stop - w.test.start == 60
        assert w.test.start > prev_start          # 时间正序
        prev_start = w.test.start


def test_walk_forward_windows_expanding_vs_rolling():
    df = _bars(n=1000)
    exp = R.walk_forward_windows(df, 4, 250, 60, 5, expanding=True)
    roll = R.walk_forward_windows(df, 4, 250, 60, 5, expanding=False)
    exp_sizes = [w.train.stop - w.train.start for w in exp]
    roll_sizes = [w.train.stop - w.train.start for w in roll]
    assert exp_sizes == sorted(exp_sizes) and exp_sizes[0] < exp_sizes[-1]
    assert len(set(roll_sizes)) == 1


def test_walk_forward_windows_stops_when_data_short():
    assert R.walk_forward_windows(_bars(n=200), n_splits=5, train_bars=250) == []


def test_walk_forward_reports_per_window(bars):
    rep = R.walk_forward(_bars(n=900), ema_cross, n_splits=4, test_bars=60)
    assert len(rep["windows"]) == 4
    for w in rep["windows"]:
        assert {"window", "test_start", "test_end", "sharpe"} <= set(w)
    assert rep["summary"]["n"] == 4


def test_walk_forward_verdict_flags_instability():
    assert "窗口太少" in R.walk_forward_verdict({"n": 2, "metric": "sharpe"})
    assert "不成立" in R.walk_forward_verdict(
        {"n": 5, "metric": "sharpe", "sharpe_median": -0.2,
         "pct_positive": 0.2, "sharpe_std": 0.3})
    assert "靠个别窗口" in R.walk_forward_verdict(
        {"n": 5, "metric": "sharpe", "sharpe_median": 0.3,
         "pct_positive": 0.4, "sharpe_std": 0.2})
    assert "波动过大" in R.walk_forward_verdict(
        {"n": 5, "metric": "sharpe", "sharpe_median": 0.3,
         "pct_positive": 0.8, "sharpe_std": 5.0})
    assert "稳定" in R.walk_forward_verdict(
        {"n": 5, "metric": "sharpe", "sharpe_median": 0.5,
         "pct_positive": 0.8, "sharpe_std": 0.3})


def test_format_walk_forward_runs():
    out = R.format_walk_forward(R.walk_forward(_bars(n=900), ema_cross, n_splits=3))
    assert "Walk-Forward" in out and "判定" in out


# ====================== 参数网格 ======================

def test_expand_grid():
    g = R.expand_grid({"a": [1, 2], "b": [10, 20, 30]})
    assert len(g) == 6
    assert {"a": 1, "b": 10} in g and {"a": 2, "b": 30} in g
    assert R.expand_grid({}) == [{}]


def test_param_sweep_sorted_and_complete(bars):
    sweep = R.param_sweep(bars, lambda fast, slow: partial(ema_cross, fast=fast, slow=slow),
                          {"fast": [3, 5], "slow": [20, 30]})
    assert len(sweep) == 4
    assert sweep["sharpe"].is_monotonic_decreasing
    assert {"fast", "slow", "sharpe", "_combo"} <= set(sweep.columns)


def test_param_sweep_rejects_empty_grid(bars):
    with pytest.raises(ValueError):
        R.param_sweep(bars, lambda: ema_cross, {"fast": []})


# ====================== 参数高原 ======================

def test_plateau_score_high_for_clustered_optima():
    """最优参数聚在一起 → 高原分高。"""
    rows = []
    for f in range(1, 11):
        for s in range(1, 11):
            # 最优区在 (5,5) 附近的一整片
            val = -((f - 5) ** 2 + (s - 5) ** 2) / 10.0
            rows.append({"fast": f, "slow": s, "sharpe": val})
    assert R.plateau_score(pd.DataFrame(rows), ["fast", "slow"]) > 0.5


def test_plateau_score_low_for_scattered_optima():
    """最优参数四散 → 高原分低（典型的拟合噪声）。"""
    rng = np.random.default_rng(0)
    rows = [{"fast": f, "slow": s, "sharpe": float(rng.normal())}
            for f in range(1, 11) for s in range(1, 11)]
    assert R.plateau_score(pd.DataFrame(rows), ["fast", "slow"]) < 0.45


def test_plateau_score_edge_cases():
    assert R.plateau_score(pd.DataFrame(), ["a"]) == 0.0
    assert R.plateau_score(pd.DataFrame({"a": [1], "sharpe": [1.0]}), []) == 0.0
    # 单一取值的参数维度（无跨度）不该崩
    df = pd.DataFrame({"a": [1, 1, 1], "sharpe": [1.0, 2.0, 3.0]})
    assert R.plateau_score(df, ["a"]) == 0.0


# ====================== 样本外优化 ======================

def test_optimize_splits_in_and_out_of_sample():
    df = _bars(n=800, seed=3, trend=0.05)
    res = R.optimize(df, lambda fast, slow: partial(ema_cross, fast=fast, slow=slow),
                     {"fast": [3, 5, 8], "slow": [20, 40]}, test_ratio=0.3)
    assert set(res["best_params"]) == {"fast", "slow"}
    assert res["n_combos"] == 6
    assert "in_sample_metric" in res and "out_of_sample_metric" in res
    assert isinstance(res["verdict"], str)


def test_optimize_preserves_param_types():
    """从 DataFrame 里读会把 int 变成 numpy.float64，
    再传给 rolling()/range() 这类只收整数的地方就会炸。"""
    df = _bars(n=800, seed=4)
    res = R.optimize(df, lambda fast, slow: partial(ema_cross, fast=fast, slow=slow),
                     {"fast": [3, 5], "slow": [20, 40]}, test_ratio=0.3)
    for v in res["best_params"].values():
        assert isinstance(v, int) and not isinstance(v, bool), f"{v!r} 类型丢了"


def test_optimize_rejects_short_series():
    with pytest.raises(ValueError, match="样本太短"):
        R.optimize(_bars(n=50), lambda fast: partial(ema_cross, fast=fast),
                   {"fast": [3, 5]}, test_ratio=0.3)


def test_optimize_verdict_covers_cases():
    assert "没跑赢" in R.optimize_verdict(-0.5, 0.5, 0.0, 0.9)
    assert "过拟合" in R.optimize_verdict(1.0, -0.5, 1.5, 0.9)
    assert "过拟合嫌疑" in R.optimize_verdict(1.0, 0.3, 0.7, 0.9)
    assert "孤立尖峰" in R.optimize_verdict(1.0, 0.9, 0.1, 0.1)
    assert "保持住" in R.optimize_verdict(1.0, 0.9, 0.1, 0.9)


def test_optimize_detects_pure_overfit():
    """在纯随机行情上寻优，样本外必然衰减——这是模块存在的意义。"""
    df = _bars(n=900, seed=99, trend=0.0, vol=2.0)
    res = R.optimize(df, lambda fast, slow: partial(ema_cross, fast=fast, slow=slow),
                     {"fast": [3, 5, 8, 12], "slow": [20, 30, 40, 60]}, test_ratio=0.35)
    # 样本内最优一定 >= 样本内中位数（选出来的就是最好的）
    assert res["in_sample_metric"] >= res["sweep"]["sharpe"].median()
    assert isinstance(res["degradation"], float)


# ====================== 随机基准 ======================

def test_random_benchmark_structure(bars):
    r = R.random_benchmark(bars, ema_cross, n_trials=20)
    assert {"actual", "random_mean", "percentile", "p_value", "verdict"} <= set(r)
    assert 0.0 <= r["p_value"] <= 1.0
    assert 0.0 <= r["percentile"] <= 100.0
    assert r["n_trials"] > 0


def test_random_benchmark_is_reproducible(bars):
    a = R.random_benchmark(bars, ema_cross, n_trials=20, seed=7)
    b = R.random_benchmark(bars, ema_cross, n_trials=20, seed=7)
    assert a["random_mean"] == pytest.approx(b["random_mean"])


def test_random_benchmark_flags_a_useless_strategy(bars):
    """永远不交易的策略，夏普 0，应该判成与随机无异（甚至更差）。"""
    flat = lambda d: pd.Series(HOLD, index=d.index)      # noqa: E731
    r = R.random_benchmark(bars, flat, n_trials=40)
    assert r["p_value"] > 0.05
    assert "随机" in r["verdict"]


def test_random_benchmark_recognises_a_clairvoyant_strategy():
    """作弊策略（用未来数据）应显著跑赢随机——验证检验本身有辨别力。"""
    df = _bars(n=400, seed=5, vol=2.0)
    fwd = df["close"].shift(-5)

    def cheat(d):
        s = pd.Series(HOLD, index=d.index)
        f = fwd.reindex(d.index)
        up = f > d["close"]
        s[up & ~up.shift(1, fill_value=False)] = BUY
        s[~up & up.shift(1, fill_value=False)] = SELL
        return s

    r = R.random_benchmark(df, cheat, n_trials=60)
    assert r["actual"] > r["random_mean"]
    assert r["p_value"] < 0.05, "有前视信息的策略必须被判为显著"


def test_config_is_respected(bars):
    """高手续费应该拉低表现——确认 cfg 真的透传下去了。"""
    cheap = R.run_once(bars, ema_cross, BacktestConfig(commission_bps=0, slippage_bps=0))
    dear = R.run_once(bars, ema_cross, BacktestConfig(commission_bps=200, slippage_bps=200))
    assert dear["total_return"] < cheap["total_return"]
