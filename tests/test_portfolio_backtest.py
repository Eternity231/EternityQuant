"""组合级回测 + ML→回测桥（v0.29）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eq.backtest.portfolio import (
    PortfolioConfig, compare_allocations, format_portfolio, run_portfolio,
)
from eq.strategy import BUY, HOLD
from eq.strategy.signals import ema_cross


def _bars(seed=0, n=300, trend=0.05, vol=1.5, start="2024-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = np.maximum(100 + np.cumsum(rng.normal(trend, vol, n)), 5.0)
    nz = np.abs(rng.normal(0, 0.8, n))
    return pd.DataFrame({"open": c, "high": c + nz, "low": c - nz, "close": c,
                         "volume": np.full(n, 1e6)},
                        index=pd.bdate_range(start, periods=n))


@pytest.fixture
def universe():
    return {f"S{i:02d}": _bars(seed=i, vol=1.0 + 0.2 * i) for i in range(8)}


def _always_long(d):
    return pd.Series(BUY, index=d.index)


# ====================== 基本运行 ======================

def test_run_portfolio_basic(universe):
    res = run_portfolio(universe, ema_cross, PortfolioConfig(initial_cash=500_000))
    assert len(res.equity_curve) > 0
    assert res.weights.shape[1] == len(universe)
    assert np.isfinite(res.metrics["total_return"])
    assert isinstance(res.summary(), str)


def test_rejects_empty_or_too_short():
    with pytest.raises(ValueError, match="足够长"):
        run_portfolio({}, ema_cross)
    with pytest.raises(ValueError, match="足够长"):
        run_portfolio({"a": _bars(n=5)}, ema_cross)


def test_weights_respect_max_positions(universe):
    cfg = PortfolioConfig(initial_cash=500_000, max_positions=3)
    res = run_portfolio(universe, _always_long, cfg)
    held = (res.weights > 1e-6).sum(axis=1)
    assert held.max() <= 3, f"最多持 3 只，实际 {held.max()}"


def test_weights_respect_max_weight(universe):
    cfg = PortfolioConfig(initial_cash=1_000_000, max_positions=8, max_weight=0.2)
    res = run_portfolio(universe, _always_long, cfg)
    # 允许一点点浮动（价格涨了权重会漂移，约束只在调仓日生效）
    assert res.weights.max().max() <= 0.35


def test_cash_buffer_leaves_cash(universe):
    cfg = PortfolioConfig(initial_cash=500_000, cash_buffer=0.20, max_positions=4)
    res = run_portfolio(universe, _always_long, cfg)
    assert res.metrics["cash_pct_mean"] > 0.1


def test_never_goes_negative_cash(universe):
    """现金不够时必须缩量，不能透支。"""
    res = run_portfolio(universe, _always_long,
                        PortfolioConfig(initial_cash=30_000, max_positions=8))
    assert (res.equity_curve >= 0).all()


# ====================== 资金分配 ======================

def test_inverse_vol_reduces_portfolio_volatility():
    """波动率反比的核心价值主张：降低组合波动。"""
    uni = {f"S{i}": _bars(seed=i, vol=0.5 + 1.2 * i) for i in range(6)}
    base = PortfolioConfig(initial_cash=500_000, max_positions=6, rebalance="weekly")
    import dataclasses
    eq = run_portfolio(uni, _always_long, dataclasses.replace(base, allocation="equal"))
    iv = run_portfolio(uni, _always_long, dataclasses.replace(base, allocation="inverse_vol"))
    assert iv.metrics["volatility"] < eq.metrics["volatility"]


def test_score_allocation_favours_strong_signals():
    uni = {f"S{i}": _bars(seed=i) for i in range(4)}
    idx = uni["S0"].index
    # S0 分数最高，S3 最低
    scores = pd.DataFrame({f"S{i}": np.full(len(idx), 1.0 - 0.25 * i) for i in range(4)},
                          index=idx)
    res = run_portfolio(uni, scores,
                        PortfolioConfig(initial_cash=500_000, max_positions=4,
                                        allocation="score", max_weight=0.9))
    mean_w = res.weights.mean()
    assert mean_w["S0"] > mean_w["S3"], "分数高的应配更多"


def test_unknown_allocation_raises(universe):
    with pytest.raises(ValueError, match="未知分配方式"):
        run_portfolio(universe, _always_long,
                      PortfolioConfig(allocation="魔法"))


def test_compare_allocations_table(universe):
    df = compare_allocations(universe, ema_cross,
                             PortfolioConfig(initial_cash=500_000, max_positions=4))
    assert len(df) == 3
    assert {"分配方式", "总收益", "夏普", "年化波动"} <= set(df.columns)


# ====================== 调仓节奏 ======================

@pytest.mark.parametrize("mode", ["signal", "daily", "weekly", "monthly"])
def test_rebalance_modes_run(universe, mode):
    res = run_portfolio(universe, ema_cross,
                        PortfolioConfig(initial_cash=500_000, rebalance=mode))
    assert np.isfinite(res.metrics["total_return"])


def test_lower_frequency_rebalance_means_lower_turnover(universe):
    import dataclasses
    base = PortfolioConfig(initial_cash=500_000, max_positions=4)
    daily = run_portfolio(universe, ema_cross, dataclasses.replace(base, rebalance="daily"))
    monthly = run_portfolio(universe, ema_cross, dataclasses.replace(base, rebalance="monthly"))
    assert monthly.metrics["annual_turnover"] <= daily.metrics["annual_turnover"]


# ====================== 分散化 ======================

def test_portfolio_volatility_below_average_component():
    """分散化收益：组合波动应低于成分股波动的平均。"""
    uni = {f"S{i}": _bars(seed=100 + i, vol=2.0) for i in range(8)}
    res = run_portfolio(uni, _always_long,
                        PortfolioConfig(initial_cash=1_000_000, max_positions=8,
                                        rebalance="monthly", cost_model="flat"))
    comp_vol = np.mean([d["close"].pct_change().std() * np.sqrt(252) for d in uni.values()])
    assert res.metrics["volatility"] < comp_vol, (
        f"组合波动 {res.metrics['volatility']:.2%} 应低于成分平均 {comp_vol:.2%}")


# ====================== 成本 / 会计 ======================

def test_costs_reduce_return(universe):
    import dataclasses
    base = PortfolioConfig(initial_cash=500_000, max_positions=4, rebalance="weekly")
    free = run_portfolio(universe, ema_cross,
                         dataclasses.replace(base, cost_model=None,
                                             commission_bps=0, slippage_bps=0))
    real = run_portfolio(universe, ema_cross, dataclasses.replace(base, cost_model="a_share"))
    assert real.metrics["total_return"] < free.metrics["total_return"]


def test_realized_plus_unrealized_reconciles(universe):
    """已实现 + 未实现 应等于总盈亏——否则逐标的贡献表会看不懂。"""
    cfg = PortfolioConfig(initial_cash=500_000, max_positions=4)
    res = run_portfolio(universe, ema_cross, cfg)
    total = res.equity_curve.iloc[-1] - cfg.initial_cash
    assert (res.metrics["realized_pnl"] + res.metrics["unrealized_pnl"]) == pytest.approx(total, rel=1e-6)


def test_contribution_table(universe):
    res = run_portfolio(universe, ema_cross, PortfolioConfig(initial_cash=500_000))
    if not res.contribution.empty:
        assert {"trades", "total_pnl", "win_rate"} <= set(res.contribution.columns)
        assert res.contribution["total_pnl"].is_monotonic_decreasing


def test_format_portfolio_runs(universe):
    out = format_portfolio(run_portfolio(universe, ema_cross,
                                         PortfolioConfig(initial_cash=500_000)))
    assert "组合回测" in out and "换手" in out


# ====================== 信号形态 ======================

def test_accepts_per_symbol_strategy_dict(universe):
    strat = {s: (_always_long if i % 2 == 0 else (lambda d: pd.Series(HOLD, index=d.index)))
             for i, s in enumerate(universe)}
    res = run_portfolio(universe, strat, PortfolioConfig(initial_cash=500_000))
    held = set(res.weights.columns[(res.weights > 1e-6).any()])
    assert held <= {s for i, s in enumerate(universe) if i % 2 == 0}


def test_score_matrix_with_no_overlap_raises(universe):
    bad = pd.DataFrame({"不存在的票": [1.0] * 50},
                       index=pd.bdate_range("2024-01-01", periods=50))
    with pytest.raises(ValueError, match="没有交集"):
        run_portfolio(universe, bad)


def test_symbols_with_different_date_ranges(universe):
    uni = dict(universe)
    uni["LATE"] = _bars(seed=99, n=120, start="2024-06-03")   # 晚上市
    res = run_portfolio(uni, _always_long, PortfolioConfig(initial_cash=500_000))
    # 上市前不能有仓位
    early = res.weights.loc[res.weights.index < pd.Timestamp("2024-06-03"), "LATE"]
    assert (early.abs() < 1e-9).all(), "标的上市前不该有持仓"


# ====================== ML 桥 ======================

def _seed_predictions(tmp_db, bars, horizon=5, noise_mult=2.0, seed=0):
    from eq.db import execute_many
    from eq.strategy.factors.ml import register_model

    mid = register_model(name="t", universe="test", features=[], algo="synthetic",
                         horizon=horizon, train_period="x")
    rng = np.random.default_rng(seed)
    rows = []
    for s, d in bars.items():
        fwd = (d["close"].shift(-horizon) / d["close"] - 1).fillna(0.0)
        score = fwd.to_numpy() + rng.normal(0, max(fwd.std(), 1e-6) * noise_mult, len(d))
        rows += [(mid, s, t.date().isoformat(), float(v))
                 for t, v in zip(d.index, score, strict=True)]
    execute_many("INSERT INTO ml_predictions (model_id,symbol,date,score) VALUES (?,?,?,?)", rows)
    return mid


def test_load_and_pivot_predictions(tmp_db, universe):
    from eq.strategy.ml_strategy import load_predictions, predictions_wide

    mid = _seed_predictions(tmp_db, universe)
    long = load_predictions(mid)
    assert len(long) == sum(len(d) for d in universe.values())
    wide = predictions_wide(mid)
    assert wide.shape[1] == len(universe)


def test_ml_score_matrix_selects_top_n(tmp_db, universe):
    from eq.strategy.ml_strategy import ml_score_matrix, predictions_wide

    mid = _seed_predictions(tmp_db, universe)
    sc = ml_score_matrix(predictions_wide(mid), top_n=3)
    nonzero = (sc > 0).sum(axis=1)
    assert nonzero.max() <= 3
    assert sc.max().max() <= 1.0 and sc.min().min() >= 0.0


def test_ml_score_matrix_hold_days_reduces_switching(tmp_db, universe):
    from eq.strategy.ml_strategy import ml_score_matrix, predictions_wide

    mid = _seed_predictions(tmp_db, universe)
    wide = predictions_wide(mid)
    daily = ml_score_matrix(wide, top_n=3, hold_days=1)
    held = ml_score_matrix(wide, top_n=3, hold_days=10)
    churn_daily = ((daily > 0) != (daily > 0).shift(1)).sum().sum()
    churn_held = ((held > 0) != (held > 0).shift(1)).sum().sum()
    assert churn_held < churn_daily, "持有期变长，换仓次数必须下降"


def test_ml_score_matrix_needs_predictions():
    from eq.strategy.ml_strategy import ml_score_matrix

    with pytest.raises(ValueError, match="没有可用的预测"):
        ml_score_matrix(pd.DataFrame())


def test_ml_signal_for_returns_signal_func(tmp_db, universe):
    from eq.strategy.ml_strategy import ml_signal_for

    mid = _seed_predictions(tmp_db, universe)
    sym = next(iter(universe))
    fn = ml_signal_for(sym, mid)
    sig = fn(universe[sym])
    assert len(sig) == len(universe[sym])
    assert set(sig.unique()) <= {"BUY", "SELL", "HOLD"}


def test_ml_signal_for_unknown_symbol_is_all_hold(tmp_db, universe):
    from eq.strategy.ml_strategy import ml_signal_for

    _seed_predictions(tmp_db, universe)
    sig = ml_signal_for("999999.SH", None)(next(iter(universe.values())))
    assert (sig == "HOLD").all()


def test_backtest_model_end_to_end(tmp_db, universe):
    from eq.strategy.ml_strategy import backtest_model

    mid = _seed_predictions(tmp_db, universe)
    out = backtest_model(mid, bars=universe, top_n=3, hold_days=5,
                         portfolio_cfg=PortfolioConfig(initial_cash=500_000,
                                                       max_positions=3,
                                                       allocation="score"))
    assert out["n_symbols"] == len(universe)
    assert np.isfinite(out["result"].metrics["total_return"])


def test_high_turnover_destroys_a_good_signal(tmp_db, universe):
    """本模块存在的理由：IC 再好，天天换手也会被成本吃光。"""
    from eq.strategy.ml_strategy import backtest_model

    mid = _seed_predictions(tmp_db, universe, noise_mult=1.0)   # 信号质量不错
    cfg = PortfolioConfig(initial_cash=500_000, max_positions=3,
                          allocation="score", cost_model="a_share")
    daily = backtest_model(mid, bars=universe, top_n=3, hold_days=1, portfolio_cfg=cfg)
    held = backtest_model(mid, bars=universe, top_n=3, hold_days=5, portfolio_cfg=cfg)
    assert daily["result"].metrics["annual_turnover"] > held["result"].metrics["annual_turnover"] * 2


def test_backtest_model_without_predictions(tmp_db):
    from eq.strategy.ml_strategy import backtest_model

    with pytest.raises(ValueError, match="没有预测记录"):
        backtest_model("不存在的模型")


# ---------- 行业集中度约束（v0.37） ----------

def _sector_bars(n_days: int = 120):
    """造 9 只票：3 个行业 × 3 只。同行业内走势相近，行业间不相关。"""
    import numpy as np
    import pandas as pd

    idx = pd.bdate_range("2024-01-01", periods=n_days)
    rng = np.random.default_rng(11)
    bars, inds = {}, {}
    for ind in ("白酒", "银行", "半导体"):
        common = np.cumsum(rng.normal(0.15, 1.0, n_days))     # 行业共同因子
        for k in range(3):
            sym = f"{ind}{k}"
            c = 100 + common + np.cumsum(rng.normal(0, 0.3, n_days))
            c = np.maximum(c, 1.0)
            bars[sym] = pd.DataFrame(
                {"open": c, "high": c * 1.01, "low": c * 0.99, "close": c,
                 "volume": 1e6},
                index=idx)
            inds[sym] = ind
    return bars, inds, idx


def _all_in_scores(bars, idx):
    """所有票每天都想买，且给同行业更高分——不加约束必然全押一个行业。"""
    import pandas as pd

    return pd.DataFrame(
        {s: (0.9 if s.startswith("白酒") else 0.5) for s in bars},
        index=idx)


def test_industry_cap_limits_same_sector_holdings():
    from eq.backtest.portfolio import PortfolioConfig, run_portfolio

    bars, inds, idx = _sector_bars()
    scores = _all_in_scores(bars, idx)

    cfg = PortfolioConfig(max_positions=6, max_per_industry=1, min_weight=0.01)
    res = run_portfolio(bars, scores, cfg, industries=inds)
    held = res.weights[res.weights > 1e-6]
    for day in held.index:
        row = held.loc[day].dropna()
        per_ind: dict[str, int] = {}
        for sym in row.index:
            per_ind[inds[sym]] = per_ind.get(inds[sym], 0) + 1
        assert all(v <= 1 for v in per_ind.values()), f"{day} 违反行业上限：{per_ind}"


def test_without_cap_portfolio_concentrates():
    """不设约束时确实会全压高分行业——证明上面的测试不是空转。"""
    from eq.backtest.portfolio import PortfolioConfig, run_portfolio

    bars, inds, idx = _sector_bars()
    res = run_portfolio(bars, _all_in_scores(bars, idx),
                        PortfolioConfig(max_positions=3, min_weight=0.01))
    last = res.weights.iloc[-1]
    holdings = [s for s in last.index if last[s] > 1e-6]
    assert holdings and all(s.startswith("白酒") for s in holdings), \
        f"无约束下应集中在白酒，实得 {holdings}"


def test_unknown_industry_is_not_blocked():
    """行业归属缺失的标的不受限——数据拿不到不该等于禁止买入。"""
    from eq.backtest.portfolio import PortfolioConfig, run_portfolio

    bars, inds, idx = _sector_bars()
    partial = {s: v for s, v in inds.items() if not s.startswith("半导体")}
    cfg = PortfolioConfig(max_positions=9, max_per_industry=1, min_weight=0.01)
    res = run_portfolio(bars, _all_in_scores(bars, idx), cfg, industries=partial)
    last = res.weights.iloc[-1]
    semis = [s for s in last.index if s.startswith("半导体") and last[s] > 1e-6]
    assert len(semis) > 1, f"未知行业的票不该被名额限制，实得 {semis}"


def test_industry_cap_off_by_default():
    from eq.backtest.portfolio import PortfolioConfig

    assert PortfolioConfig().max_per_industry == 0


def test_industry_cap_ignored_without_mapping():
    """只设 max_per_industry 但没给 industries 时按不限制处理，不能报错。"""
    from eq.backtest.portfolio import PortfolioConfig, run_portfolio

    bars, _, idx = _sector_bars()
    res = run_portfolio(bars, _all_in_scores(bars, idx),
                        PortfolioConfig(max_positions=3, max_per_industry=1,
                                        min_weight=0.01))
    assert len(res.equity_curve) > 0
