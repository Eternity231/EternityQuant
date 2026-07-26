"""本地 Alpha158（v0.39）。

qlib 没装、也没有 .bin 数据，所以**不是**和 qlib 逐位对拍，而是验证：

1. 每个算子算得对（用手算得出闭式解的构造数据）
2. **没有前视**——这是最关键的一条，特征只能用当期及历史
3. 去量纲有效（价格/成交量差几个数量级的票，特征值仍可比）
4. 边界不炸（停牌零成交量、常数序列、超短历史）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eq.strategy.factors import alpha as A


def _bars(n=120, seed=0, start=100.0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    c = start * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    o = c * (1 + rng.normal(0, 0.004, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.006, n)))
    v = rng.integers(1e6, 9e6, n).astype(float)
    return pd.DataFrame({"open": o, "high": h, "low": low, "close": c,
                         "volume": v, "amount": v * c}, index=idx)


# ---------- 特征清单 ----------

def test_produces_158_features():
    names = A.feature_names()
    assert len(names) == 158, f"应为 158 个特征，实得 {len(names)}"
    assert len(set(names)) == 158, "不允许重名"


def test_group_sizes():
    """9 个 KBAR + 4 个价格 + 29 个滚动 × 5 个窗口 = 158。"""
    names = A.feature_names()
    kbar = [n for n in names if n in
            {"KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2"}]
    price = [n for n in names if n in {"OPEN0", "HIGH0", "LOW0", "VWAP0"}]
    assert len(kbar) == 9 and len(price) == 4
    assert len(names) - 13 == 29 * len(A.WINDOWS)


def test_all_windows_present():
    names = set(A.feature_names())
    for d in A.WINDOWS:
        for stem in ("ROC", "MA", "STD", "RSV", "CORR", "WVMA", "VSUMD"):
            assert f"{stem}{d}" in names, f"缺 {stem}{d}"


# ---------- 前视：最关键的一条 ----------

def test_no_lookahead_appending_future_bars_changes_nothing():
    """在序列尾部追加新数据，**已有行的特征值必须一字不变**。

    这是检测前视最直接的办法：如果某个特征偷看了未来，
    加了未来数据之后历史行的取值就会变。
    """
    df = _bars(120, seed=1)
    early = A.alpha158_single(df.iloc[:100])
    full = A.alpha158_single(df)
    common = full.iloc[:100]
    pd.testing.assert_frame_equal(early, common, check_exact=False, atol=1e-12,
                                  obj="追加未来数据后历史特征被改写＝前视")


def test_no_lookahead_modifying_future_does_not_change_past():
    """把最后 20 根 bar 改成天文数字，前 100 行也不该动。"""
    df = _bars(120, seed=2)
    a = A.alpha158_single(df).iloc[:100]
    tampered = df.copy()
    tampered.iloc[100:] *= 1000
    b = A.alpha158_single(tampered).iloc[:100]
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-12)


def test_forward_return_is_the_only_forward_looking_thing():
    """标签当然要看未来，但末尾 h 根必须是 NaN（未来还没发生）。"""
    bars = {"A": _bars(50)}
    r = A.forward_return(bars, horizon=5)
    assert r.isna().sum() == 5
    assert r.notna().sum() == 45


# ---------- 算子正确性（构造出能手算的数据） ----------

def test_ma_on_linear_series():
    """等差数列上 MA(d) 的闭式解：均值 = 当前值 - (d-1)/2 × 步长。"""
    idx = pd.bdate_range("2024-01-01", periods=30)
    c = pd.Series(np.arange(100.0, 130.0), index=idx)
    df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e6})
    out = A.alpha158_single(df, windows=(5,))
    last_c = c.iloc[-1]
    assert out["MA5"].iloc[-1] == pytest.approx((last_c - 2.0) / last_c)


def test_slope_beta_on_linear_series():
    """严格线性时回归斜率就是步长，R² 应为 1，残差为 0。"""
    idx = pd.bdate_range("2024-01-01", periods=40)
    c = pd.Series(np.arange(100.0, 140.0) * 1.0, index=idx)
    df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e6})
    out = A.alpha158_single(df, windows=(10,))
    assert out["BETA10"].iloc[-1] == pytest.approx(1.0 / c.iloc[-1], rel=1e-9)
    assert out["RSQR10"].iloc[-1] == pytest.approx(1.0, abs=1e-9)
    assert out["RESI10"].iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_rsquare_is_zero_on_pure_noise_ish():
    """噪声上 R² 该远小于 1（不做精确断言，只钉住量级）。"""
    df = _bars(200, seed=5)
    out = A.alpha158_single(df, windows=(60,))
    assert out["RSQR60"].dropna().mean() < 0.7


def test_rsv_bounds():
    """RSV = (close-最低)/(最高-最低)，必须落在 [0,1]。"""
    out = A.alpha158_single(_bars(150, seed=3))
    for d in A.WINDOWS:
        s = out[f"RSV{d}"].dropna()
        assert s.min() >= -1e-9 and s.max() <= 1 + 1e-9, f"RSV{d} 越界"


def test_idx_max_is_days_since_high():
    """今天创新高 → IMAX=0；高点在窗口最左端 → IMAX=(d-1)/d。"""
    idx = pd.bdate_range("2024-01-01", periods=20)
    h = pd.Series([1.0] * 19 + [99.0], index=idx)      # 今天最高
    df = pd.DataFrame({"open": h, "high": h, "low": h, "close": h, "volume": 1e6})
    assert A.alpha158_single(df, windows=(5,))["IMAX5"].iloc[-1] == pytest.approx(0.0)

    h2 = pd.Series([1.0] * 20, index=idx)
    h2.iloc[-5] = 99.0                                  # 高点在 5 日窗口最左端
    df2 = pd.DataFrame({"open": h2, "high": h2, "low": h2, "close": h2, "volume": 1e6})
    assert A.alpha158_single(df2, windows=(5,))["IMAX5"].iloc[-1] == pytest.approx(4 / 5)


def test_cntp_counts_up_days():
    """连涨序列的 CNTP 应为 1，CNTN 为 0。"""
    idx = pd.bdate_range("2024-01-01", periods=30)
    c = pd.Series(np.arange(100.0, 130.0), index=idx)
    df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e6})
    out = A.alpha158_single(df, windows=(10,))
    assert out["CNTP10"].iloc[-1] == pytest.approx(1.0)
    assert out["CNTN10"].iloc[-1] == pytest.approx(0.0)
    assert out["CNTD10"].iloc[-1] == pytest.approx(1.0)


def test_sump_on_monotonic_series():
    """只涨不跌时 SUMP=1、SUMN=0、SUMD=1。"""
    idx = pd.bdate_range("2024-01-01", periods=30)
    c = pd.Series(np.arange(100.0, 130.0), index=idx)
    df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e6})
    out = A.alpha158_single(df, windows=(10,))
    assert out["SUMP10"].iloc[-1] == pytest.approx(1.0, abs=1e-9)
    assert out["SUMD10"].iloc[-1] == pytest.approx(1.0, abs=1e-9)


def test_rank_pct_bounds():
    out = A.alpha158_single(_bars(150, seed=7))
    s = out["RANK20"].dropna()
    assert s.min() > 0 and s.max() <= 1.0


def test_kbar_shapes():
    """一根「光头光脚大阳线」：上下影线为 0，KMID 等于全幅涨幅。"""
    idx = pd.bdate_range("2024-01-01", periods=70)
    o = pd.Series(10.0, index=idx)
    c = pd.Series(11.0, index=idx)
    df = pd.DataFrame({"open": o, "high": c, "low": o, "close": c, "volume": 1e6})
    out = A.alpha158_single(df)
    assert out["KMID"].iloc[-1] == pytest.approx(0.1)
    assert out["KUP"].iloc[-1] == pytest.approx(0.0)
    assert out["KLOW"].iloc[-1] == pytest.approx(0.0)
    assert out["KLEN"].iloc[-1] == pytest.approx(0.1)


# ---------- 去量纲：能不能放进同一个截面 ----------

def test_features_are_scale_invariant():
    """把价格和成交量整体放大 1000 倍，特征值应几乎不变。

    这是「能不能把 5 元的票和 500 元的票放进同一个截面」的前提。
    做不到的话模型会先学会区分大小盘，那不是 alpha。
    """
    df = _bars(150, seed=11)
    a = A.alpha158_single(df)
    scaled = df.copy()
    for col in ("open", "high", "low", "close"):
        scaled[col] *= 1000
    scaled["volume"] *= 1000
    scaled["amount"] = scaled["volume"] * scaled["close"]
    b = A.alpha158_single(scaled)

    bad = []
    for col in a.columns:
        x, y = a[col].dropna(), b[col].dropna()
        idx = x.index.intersection(y.index)
        if len(idx) == 0:
            continue
        d = (x[idx] - y[idx]).abs().max()
        scale = max(1.0, float(x[idx].abs().max()))
        if d / scale > 1e-6:
            bad.append((col, float(d)))
    assert not bad, f"这些特征没做到尺度不变：{bad[:8]}"


# ---------- 边界 ----------

def test_zero_volume_does_not_produce_inf():
    """停牌日成交量为 0，量类特征的分母会变成 0。"""
    df = _bars(120, seed=13)
    df.iloc[50:55, df.columns.get_loc("volume")] = 0.0
    out = A.alpha158_single(df)
    assert not np.isinf(out.to_numpy()).any(), "不允许出现 inf"


def test_constant_series_is_finite():
    """一字板停牌：OHLC 全相等、无波动。"""
    idx = pd.bdate_range("2024-01-01", periods=100)
    flat = pd.Series(10.0, index=idx)
    df = pd.DataFrame({"open": flat, "high": flat, "low": flat, "close": flat,
                       "volume": pd.Series(1e6, index=idx)})
    out = A.alpha158_single(df)
    assert not np.isinf(out.to_numpy()).any()


def test_missing_required_column_raises():
    df = _bars(80).drop(columns=["volume"])
    with pytest.raises(ValueError, match="缺少必需列"):
        A.alpha158_single(df)


def test_vwap_fallback_without_amount():
    """没有 amount 列时用典型价兜底，而不是丢掉这个特征。"""
    df = _bars(80).drop(columns=["amount"])
    out = A.alpha158_single(df)
    assert "VWAP0" in out.columns and out["VWAP0"].notna().any()


# ---------- 面板 ----------

def test_panel_shape_and_index():
    bars = {f"S{i}": _bars(120, seed=i) for i in range(4)}
    panel = A.alpha158(bars)
    assert list(panel.index.names) == ["datetime", "instrument"]
    assert panel.shape[1] == 158
    assert panel.index.get_level_values("instrument").nunique() == 4


def test_panel_skips_too_short():
    bars = {"long": _bars(120), "short": _bars(20)}
    panel = A.alpha158(bars)
    assert set(panel.index.get_level_values("instrument")) == {"long"}


def test_panel_empty_input_returns_typed_empty():
    panel = A.alpha158({})
    assert panel.empty and len(panel.columns) == 158
    assert list(panel.index.names) == ["datetime", "instrument"]


def test_panel_survives_one_bad_symbol():
    """一只票算崩不该拖垮整个面板。"""
    bars = {"good": _bars(120), "bad": _bars(120).drop(columns=["high"])}
    panel = A.alpha158(bars)
    assert set(panel.index.get_level_values("instrument")) == {"good"}


# ---------- 和下游对接 ----------

def test_feeds_preprocess_pipeline():
    from eq.strategy.factors import preprocess as pp

    bars = {f"S{i:02d}": _bars(140, seed=i) for i in range(12)}
    panel = A.alpha158(bars).dropna(how="all")
    out = pp.default_pipeline().fit_transform(panel)
    assert np.isfinite(out.to_numpy()).all(), "预处理后必须全是有限值"
    assert out.shape == panel.shape


def test_end_to_end_trains_a_model():
    """全链路：OHLCV → 特征 → 预处理 → LightGBM，完全不经 qlib。"""
    pytest.importorskip("lightgbm")
    from eq.strategy.factors import preprocess as pp
    from eq.strategy.factors.gbdt import train_gbdt

    bars = {f"S{i:02d}": _bars(200, seed=100 + i) for i in range(15)}
    x = A.alpha158(bars)
    y = A.forward_return(bars, horizon=5)
    x, y = x.align(y, join="inner", axis=0)
    x, y = pp.dropna_label(x, y)

    days = x.index.get_level_values("datetime").unique()
    cut = days[int(len(days) * 0.7)]
    m = x.index.get_level_values("datetime") < cut
    pipe = pp.default_pipeline().fit(x[m])
    xt, xv = pipe.transform(x[m]), pipe.transform(x[~m])
    yt = pp.normalize_label(y[m], "rank")
    yv = pp.normalize_label(y[~m], "rank")

    model = train_gbdt(xt, yt, xv, yv,
                       params={"num_leaves": 7, "min_data_in_leaf": 20,
                               "lambda_l1": 0.0, "lambda_l2": 0.0},
                       num_boost_round=30, early_stopping_rounds=10)
    pred = model.predict(xv)
    assert len(pred) == len(xv) and np.isfinite(pred).all()


def test_compare_with_qlib_reports_absence_gracefully():
    out = A.compare_with_qlib("600519.SH", "2024-01-01", "2024-06-30")
    assert "error" in out or "matched" in out
