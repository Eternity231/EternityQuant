"""因子评估指标（v0.25 新增，纯逻辑无网络无 GPU）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eq.strategy.factors import evaluation as ev


def _panel(n_days=60, n_stocks=50, ic_strength=0.0, seed=0):
    """造一个 (datetime, instrument) 面板：label 与 pred 的横截面相关约为 ic_strength。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    idx = pd.MultiIndex.from_product([dates, [f"S{i:03d}" for i in range(n_stocks)]],
                                     names=["datetime", "instrument"])
    pred = rng.normal(size=len(idx))
    noise = rng.normal(size=len(idx))
    label = ic_strength * pred + np.sqrt(max(1e-9, 1 - ic_strength ** 2)) * noise
    return pd.Series(pred, index=idx, name="pred"), pd.Series(label, index=idx, name="label")


# ---------- daily_ic ----------

def test_daily_ic_returns_one_value_per_day():
    p, lb = _panel(n_days=40, ic_strength=0.3)
    ic = ev.daily_ic(p, lb)
    assert len(ic) == 40
    assert isinstance(ic.index, pd.DatetimeIndex)


def test_daily_ic_recovers_known_signal():
    p, lb = _panel(n_days=120, n_stocks=200, ic_strength=0.30, seed=1)
    ic = ev.daily_ic(p, lb)
    # Spearman 对线性高斯关系略低于 Pearson，给一个宽区间
    assert 0.20 < ic.mean() < 0.35


def test_daily_ic_near_zero_for_pure_noise():
    p, lb = _panel(n_days=120, n_stocks=200, ic_strength=0.0, seed=2)
    assert abs(ev.daily_ic(p, lb).mean()) < 0.03


def test_daily_ic_detects_sign_flip():
    p, lb = _panel(n_days=80, n_stocks=100, ic_strength=0.25, seed=3)
    assert ev.daily_ic(p, -lb).mean() < -0.15


def test_daily_ic_skips_days_with_too_few_stocks():
    p, lb = _panel(n_days=10, n_stocks=3)
    assert len(ev.daily_ic(p, lb, min_stocks=5)) == 0
    assert len(ev.daily_ic(p, lb, min_stocks=2)) == 10


def test_daily_ic_constant_prediction_scores_zero_not_nan():
    """模型输出塌缩成常数是常见故障，必须显性记 0 而不是被 dropna 藏掉。"""
    p, lb = _panel(n_days=20, n_stocks=50)
    flat = pd.Series(np.ones(len(p)), index=p.index)
    ic = ev.daily_ic(flat, lb)
    assert len(ic) == 20
    assert (ic == 0).all()


def test_daily_ic_pooled_fallback_without_dates():
    """没有日期索引时降级为 pooled，返回单元素序列。"""
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    ic = ev.daily_ic(a, a * 2 + 0.01 * rng.normal(size=500))
    assert len(ic) == 1
    assert ic.iloc[0] > 0.9


def test_daily_ic_drops_nan_and_inf():
    p, lb = _panel(n_days=10, n_stocks=30)
    p = p.copy()
    p.iloc[:5] = np.nan
    p.iloc[5:8] = np.inf
    ic = ev.daily_ic(p, lb)
    assert len(ic) == 10
    assert ic.notna().all()


# ---------- ic_summary ----------

def test_ic_summary_math():
    ic = pd.Series([0.05, 0.03, -0.01, 0.04, 0.02])
    s = ev.ic_summary(ic)
    assert s["ic_mean"] == pytest.approx(0.026)
    assert s["icir"] == pytest.approx(0.026 / ic.std(ddof=1))
    assert s["t_stat"] == pytest.approx(s["icir"] * np.sqrt(5))
    assert s["ic_win_rate"] == pytest.approx(0.8)
    assert s["n_days"] == 5


def test_ic_summary_empty():
    s = ev.ic_summary(pd.Series(dtype=float))
    assert s["ic_mean"] == 0.0 and s["n_days"] == 0


def test_ic_summary_zero_variance():
    s = ev.ic_summary(pd.Series([0.02] * 10))
    assert s["icir"] == 0.0 and s["t_stat"] == 0.0


# ---------- quantile_returns ----------

def test_quantile_returns_monotonic_for_real_signal():
    p, lb = _panel(n_days=120, n_stocks=200, ic_strength=0.35, seed=4)
    q = ev.quantile_returns(p, lb, n_groups=5)
    assert len(q["group_mean"]) == 5
    assert q["long_short"] > 0
    assert q["monotonic"] is True
    assert q["group_mean"][-1] > q["group_mean"][0]


def test_quantile_returns_flat_for_noise():
    p, lb = _panel(n_days=120, n_stocks=200, ic_strength=0.0, seed=5)
    q = ev.quantile_returns(p, lb, n_groups=5)
    assert abs(q["long_short"]) < 0.05


def test_quantile_returns_skips_thin_days():
    p, lb = _panel(n_days=10, n_stocks=3)
    assert ev.quantile_returns(p, lb, n_groups=5)["n_days"] == 0


# ---------- evaluate / 报告 ----------

def test_evaluate_full_report():
    p, lb = _panel(n_days=120, n_stocks=150, ic_strength=0.3, seed=6)
    r = ev.evaluate(p, lb)
    for k in ("ic_mean", "ic_std", "icir", "t_stat", "ic_win_rate",
              "group_mean", "long_short", "monotonic", "pooled", "n_samples"):
        assert k in r, k
    assert r["pooled"] is False
    assert r["n_samples"] == 120 * 150
    assert r["ic_mean"] > 0.2


def test_evaluate_marks_pooled_without_dates():
    rng = np.random.default_rng(0)
    r = ev.evaluate(rng.normal(size=300), rng.normal(size=300))
    assert r["pooled"] is True


def test_format_report_runs():
    p, lb = _panel(n_days=60, ic_strength=0.25, seed=7)
    out = ev.format_report(ev.evaluate(p, lb), title="单测")
    assert "Rank IC" in out and "ICIR" in out and "判定" in out


@pytest.mark.parametrize(("ic_strength", "expect_keyword"), [
    (0.0, "不显著"),
    (0.35, "可用"),
])
def test_verdict_discriminates(ic_strength, expect_keyword):
    p, lb = _panel(n_days=150, n_stocks=200, ic_strength=ic_strength, seed=8)
    assert expect_keyword in ev.verdict(ev.evaluate(p, lb))


def test_verdict_flags_short_sample():
    p, lb = _panel(n_days=10, n_stocks=50, ic_strength=0.3)
    assert "太短" in ev.verdict(ev.evaluate(p, lb))


def test_verdict_flags_negative_ic():
    p, lb = _panel(n_days=150, n_stocks=200, ic_strength=0.35, seed=9)
    assert "方向" in ev.verdict(ev.evaluate(p, -lb))


def test_pooled_ic_is_inflated_vs_daily_ic():
    """本模块存在的理由：pooled 口径会被日间漂移带偏，按日口径才是真信号。

    造一个「完全不会选股、只会跟随每日市场均值」的预测：
    横截面上毫无区分度（每日 IC≈0），但 pooled 相关很高。
    """
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2024-01-01", periods=120)
    stocks = [f"S{i:03d}" for i in range(80)]
    idx = pd.MultiIndex.from_product([dates, stocks], names=["datetime", "instrument"])
    market = pd.Series(rng.normal(0, 0.05, len(dates)), index=dates)
    beta_part = market.reindex(idx.get_level_values(0)).to_numpy()

    label = pd.Series(beta_part + rng.normal(0, 0.01, len(idx)), index=idx)
    pred = pd.Series(beta_part, index=idx)          # 只预测大盘，零选股能力

    pooled = pred.corr(label)                        # 老口径
    daily = ev.daily_ic(pred, label).mean()          # 新口径
    assert pooled > 0.9, "pooled 口径把无选股能力的模型评得很高"
    assert abs(daily) < 0.1, "按日横截面口径正确地判定它没有选股能力"


# ---------- 重叠标签的 t 值修正（v0.40） ----------

def test_newey_west_shrinks_t_under_overlap():
    """重叠标签造成正自相关时，修正后的 t 必须明显变小。

    horizon=5 的标签是 close[t+5]/close[t]-1，相邻交易日共用 4 天行情。
    普通 t = ICIR×√n 把这些天当独立样本，系统性高估显著性——
    粗略地说高估 √horizon 倍。
    """
    import numpy as np
    import pandas as pd

    from eq.strategy.factors.evaluation import ic_summary

    rng = np.random.default_rng(0)
    raw = rng.normal(0.02, 0.2, 400)
    overlapped = pd.Series(pd.Series(raw).rolling(5).mean().dropna().to_numpy())
    plain = ic_summary(overlapped, horizon=1)["t_stat"]
    fixed = ic_summary(overlapped, horizon=5)["t_stat_nw"]
    assert abs(fixed) < abs(plain) * 0.75, f"{plain:.2f} → {fixed:.2f} 缩得不够"


def test_newey_west_is_harmless_without_autocorrelation():
    """序列本来就独立时，传 horizon>1 不该无故压低 t 值。"""
    import numpy as np
    import pandas as pd

    from eq.strategy.factors.evaluation import ic_summary

    s = pd.Series(np.random.default_rng(1).normal(0.02, 0.2, 400))
    r = ic_summary(s, horizon=5)
    assert abs(r["t_stat_nw"] - r["t_stat"]) < 0.3


def test_horizon_one_falls_back_to_plain_t():
    import numpy as np
    import pandas as pd

    from eq.strategy.factors.evaluation import ic_summary

    s = pd.Series(np.random.default_rng(2).normal(0.02, 0.2, 200))
    r = ic_summary(s, horizon=1)
    assert r["t_stat_nw"] == pytest.approx(r["t_stat"], rel=1e-9)


def test_empty_ic_has_zero_nw_t():
    import pandas as pd

    from eq.strategy.factors.evaluation import ic_summary

    assert ic_summary(pd.Series(dtype=float), horizon=5)["t_stat_nw"] == 0.0


def test_evaluate_passes_horizon_through():
    import numpy as np
    import pandas as pd

    from eq.strategy.factors.evaluation import evaluate

    rng = np.random.default_rng(3)
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range("2024-01-01", periods=80), [f"S{i:02d}" for i in range(20)]],
        names=["datetime", "instrument"])
    lab = pd.Series(rng.normal(size=len(idx)), index=idx)
    pred = lab + rng.normal(scale=0.5, size=len(idx))
    r1 = evaluate(pred, lab, horizon=1)
    r5 = evaluate(pred, lab, horizon=5)
    assert r1["t_stat"] == pytest.approx(r5["t_stat"]), "普通 t 不受 horizon 影响"
    assert r1["t_stat_nw"] != r5["t_stat_nw"], "修正 t 必须随 horizon 变"
