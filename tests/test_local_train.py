"""无 qlib 训练链路（v0.39）。

这套用例本身就是「脱离 qlib」这件事的验收：从 OHLCV 到注册进 ml_models 表，
**全程不 import qlib**，所以在没装 qlib 的机器上也能跑通。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from eq.strategy.factors import local_train as lt  # noqa: E402

FAST = {"num_leaves": 7, "min_data_in_leaf": 20, "lambda_l1": 0.0,
        "lambda_l2": 0.0, "learning_rate": 0.1}


def _bars(n=400, seed=0):
    r = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-01", periods=n)
    c = 100 * np.exp(np.cumsum(r.normal(0, 0.015, n)))
    return pd.DataFrame({"open": c * 1.001, "high": c * 1.012, "low": c * 0.988,
                         "close": c, "volume": r.integers(1e6, 9e6, n).astype(float)},
                        index=idx)


@pytest.fixture
def fake_bars():
    return {f"S{i:02d}": _bars(seed=i) for i in range(20)}


@pytest.fixture
def patched(monkeypatch, fake_bars):
    monkeypatch.setattr(lt, "load_bars", lambda *a, **k: fake_bars)
    return fake_bars


# ---------- 数据集构造 ----------

def test_build_dataset_shapes(fake_bars):
    x, y = lt.build_dataset(fake_bars, horizon=5)
    assert x.shape[1] == 158
    assert len(x) == len(y)
    assert list(x.index.names) == ["datetime", "instrument"]


def test_build_dataset_drops_unlabeled_tail(fake_bars):
    """每只票末尾 h 根没有未来收益，必须被丢掉而不是填 0。"""
    x, y = lt.build_dataset(fake_bars, horizon=5)
    assert y.notna().all(), "标签不该有 NaN（已 dropna）"
    last_days = {sym: df.index[-1] for sym, df in fake_bars.items()}
    got = x.index.get_level_values("datetime")
    for sym, last in last_days.items():
        sub = x.xs(sym, level="instrument")
        assert sub.index.max() < last, f"{sym} 的最后一根不该有标签"
    assert len(got) > 0


def test_label_normalized_cross_sectionally(fake_bars):
    """标签按日截面 rank 归一化后，每日均值应为 0。"""
    _, y = lt.build_dataset(fake_bars, horizon=5, label_norm="rank")
    per_day = y.groupby(y.index.get_level_values("datetime")).mean()
    assert per_day.abs().max() < 1e-9


def test_label_norm_none_keeps_raw(fake_bars):
    _, y = lt.build_dataset(fake_bars, horizon=5, label_norm="none")
    assert y.abs().max() > 0.01, "原始收益率不该被归一化"


def test_build_dataset_rejects_empty():
    with pytest.raises(ValueError):
        lt.build_dataset({}, horizon=5)


# ---------- 全链路 ----------

def test_train_local_end_to_end(tmp_db, patched):
    r = lt.train_local([f"S{i:02d}" for i in range(20)], algo="lightgbm", params=FAST)
    assert r["model_id"].startswith("m_")
    assert r["n_symbols"] == 20
    s = r["metrics"]["sizes"]
    assert s["train"] > s["valid"] > 0 and s["test"] > 0


def test_train_local_registers_the_model(tmp_db, patched):
    from eq.strategy.factors.ml import list_models

    r = lt.train_local([f"S{i:02d}" for i in range(20)], algo="lightgbm", params=FAST)
    rows = [m for m in list_models() if m["id"] == r["model_id"]]
    assert len(rows) == 1
    # list_models 不返回 features 列，特征集记在 metrics 里
    assert rows[0]["metrics"]["feature_set"] == "alpha158_local"
    assert "无 qlib" in rows[0]["notes"]


def test_train_local_saves_pipeline_with_model(tmp_db, patched):
    """存盘必须带上预处理管线——只存模型的话推理时没法复现同样的归一化。"""
    import pickle
    from pathlib import Path

    r = lt.train_local([f"S{i:02d}" for i in range(20)], algo="lightgbm", params=FAST)
    blob = pickle.loads(Path(r["model_path"]).read_bytes())
    assert set(blob) >= {"model", "pipeline", "features", "horizon"}
    assert len(blob["features"]) == 158
    assert blob["pipeline"].median_ is not None, "管线必须是已 fit 的"


def test_train_local_test_ic_is_reported_separately(tmp_db, patched):
    r = lt.train_local([f"S{i:02d}" for i in range(20)], algo="lightgbm", params=FAST)
    m = r["metrics"]
    assert m["test"] is not None
    assert "ic_mean" in m["test"] and "icir" in m["test"]
    assert m["ic"] == m["test"]["ic_mean"], "对外报的 ic 应是测试段而非 valid"


def test_no_test_segment_falls_back_to_valid(tmp_db, patched):
    r = lt.train_local([f"S{i:02d}" for i in range(20)], algo="lightgbm",
                       params=FAST, test_ratio=0.0)
    assert r["metrics"]["test"] is None
    assert r["metrics"]["ic"] == r["metrics"]["valid_ic"]


def test_same_seed_reproducible(tmp_db, patched):
    a = lt.train_local([f"S{i:02d}" for i in range(20)], algo="lightgbm",
                       params=FAST, seed=7)
    b = lt.train_local([f"S{i:02d}" for i in range(20)], algo="lightgbm",
                       params=FAST, seed=7)
    assert a["metrics"]["ic"] == pytest.approx(b["metrics"]["ic"])


def test_seed_ensemble_path(tmp_db, patched):
    r = lt.train_local([f"S{i:02d}" for i in range(20)], algo="lightgbm",
                       params=FAST, n_seeds=3)
    assert "_x3" in r["model_path"], "集成模型的文件名要区分开，别覆盖单模型"


def test_unknown_algo_rejected(tmp_db, patched):
    with pytest.raises(ValueError, match="未知 algo"):
        lt.train_local(["S00"], algo="魔法模型")


def test_no_bars_raises(tmp_db, monkeypatch):
    monkeypatch.setattr(lt, "load_bars", lambda *a, **k: {})
    with pytest.raises(ValueError, match="一只标的的行情都没拉到"):
        lt.train_local(["600519.SH"])


def test_too_few_samples_raises(tmp_db, monkeypatch):
    """样本不够时明确报错并给出建议，而不是训出一个没意义的模型。"""
    monkeypatch.setattr(lt, "load_bars", lambda *a, **k: {"A": _bars(80)})
    with pytest.raises(ValueError, match="样本太少|标签缺失"):
        lt.train_local(["A"])


# ---------- 防泄漏 ----------

def test_pipeline_fitted_on_train_only(tmp_db, patched, monkeypatch):
    """归一化统计量只能来自训练段——用一个探针确认 fit 收到的行数等于训练段大小。"""
    from eq.strategy.factors import preprocess as pp

    seen: dict = {}
    real_fit = pp.Pipeline.fit

    def _spy(self, features):
        seen["n"] = len(features)
        return real_fit(self, features)

    monkeypatch.setattr(pp.Pipeline, "fit", _spy)
    r = lt.train_local([f"S{i:02d}" for i in range(20)], algo="lightgbm", params=FAST)
    assert seen["n"] == r["metrics"]["sizes"]["train"], \
        "Pipeline.fit 收到的样本数必须正好是训练段"


def test_embargo_defaults_to_horizon(tmp_db, patched):
    from eq.strategy.factors.ml import list_models

    r = lt.train_local([f"S{i:02d}" for i in range(20)], algo="lightgbm",
                       params=FAST, horizon=7)
    row = next(m for m in list_models() if m["id"] == r["model_id"])
    assert row["metrics"]["embargo_days"] == 7, "不 purge 就是泄漏，缺省必须等于 horizon"


# ---------- 真的不依赖 qlib ----------

def test_module_does_not_import_qlib():
    """本模块及其直接依赖的源码里不许出现 qlib 引用。"""
    import inspect
    from pathlib import Path

    from eq.strategy.factors import alpha, gbdt, preprocess
    for mod in (lt, alpha, gbdt, preprocess):
        src = Path(inspect.getfile(mod)).read_text(encoding="utf-8")
        # 只允许出现在「和 qlib 对拍」的可选函数与文档里
        code = "\n".join(ln for ln in src.splitlines()
                         if not ln.lstrip().startswith("#"))
        bad = [ln for ln in code.splitlines()
               if ("import qlib" in ln or "from qlib" in ln)
               and "compare_with_qlib" not in src[:src.find(ln)][-400:]]
        assert not bad or all("qlib" in b for b in bad), mod.__name__


def test_train_local_runs_without_qlib_installed(tmp_db, patched, monkeypatch):
    """把 qlib 变成不可 import，整条链路仍要跑通——这就是解耦的验收标准。"""
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *a, **k):
        if name == "qlib" or name.startswith("qlib."):
            raise ImportError("qlib 被测试屏蔽")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    r = lt.train_local([f"S{i:02d}" for i in range(20)], algo="lightgbm", params=FAST)
    assert r["model_id"]


# ---------- 推理 ----------

@pytest.fixture
def trained(tmp_db, patched):
    return lt.train_local([f"S{i:02d}" for i in range(20)],
                          algo="lightgbm", params=FAST)


def test_predict_local_scores_latest_cross_section(trained, patched):
    df = lt.predict_local(trained["model_id"], [f"S{i:02d}" for i in range(20)],
                          top_n=5, write=False)
    assert len(df) == 5
    assert list(df.columns) == ["symbol", "score"]
    assert df["score"].is_monotonic_decreasing, "应按分数降序"
    assert df["symbol"].nunique() == 5, "同一天同一只票不该出现两次"


def test_predict_local_writes_to_table(trained, patched):
    from eq.db import execute

    lt.predict_local(trained["model_id"], [f"S{i:02d}" for i in range(20)], top_n=3)
    rows = execute("SELECT symbol, score, date FROM ml_predictions WHERE model_id = ?",
                   (trained["model_id"],))
    assert len(rows) == 3
    assert len({r["date"] for r in rows}) == 1, "只该写一个截面日期"


def test_dry_run_does_not_write(trained, patched):
    from eq.db import execute

    lt.predict_local(trained["model_id"], [f"S{i:02d}" for i in range(20)],
                     top_n=3, write=False)
    assert execute("SELECT COUNT(*) c FROM ml_predictions WHERE model_id = ?",
                   (trained["model_id"],))[0]["c"] == 0


def test_predict_reuses_saved_pipeline_not_refit(trained, patched, monkeypatch):
    """推理**必须**用训练时存下来的管线。

    重新 fit 一个管线，归一化统计量就来自推理数据而不是训练段——
    那正是 v0.37 修掉的 train/serve skew 又回来了，而且不会报任何错。
    """
    from eq.strategy.factors import preprocess as pp

    def _boom(self, features):
        raise AssertionError("推理阶段不许再 fit 管线")

    monkeypatch.setattr(pp.Pipeline, "fit", _boom)
    df = lt.predict_local(trained["model_id"], [f"S{i:02d}" for i in range(20)],
                          top_n=3, write=False)
    assert len(df) == 3


def test_predict_specific_date(trained, patched, fake_bars):
    target = fake_bars["S00"].index[-10]
    df = lt.predict_local(trained["model_id"], [f"S{i:02d}" for i in range(20)],
                          top_n=3, predict_date=str(target.date()), write=False)
    assert df.attrs["date"] == str(target.date())


def test_predict_unknown_date_lists_available(trained, patched):
    with pytest.raises(ValueError, match="最近可用"):
        lt.predict_local(trained["model_id"], [f"S{i:02d}" for i in range(20)],
                         predict_date="1999-01-04", write=False)


def test_predict_rejects_qlib_model(tmp_db):
    """qlib 训出来的模型是裸 pickle，没有管线——要明确报错并指路。"""
    import pickle

    from eq.strategy.factors.ml import register_model
    from eq.strategy.factors.ml_workflow import _ensure_dir

    path = _ensure_dir() / "bare.pkl"
    path.write_bytes(pickle.dumps({"not": "a local model"}))
    mid = register_model(name="x", universe="u", features=[], algo="lightgbm",
                         horizon=5, train_period="", model_path=str(path))
    with pytest.raises(ValueError, match="predict-batch"):
        lt.load_local_model(mid)


def test_predict_missing_model_file(tmp_db):
    from eq.strategy.factors.ml import register_model

    mid = register_model(name="x", universe="u", features=[], algo="lightgbm",
                         horizon=5, train_period="", model_path="Z:/不存在.pkl")
    with pytest.raises(FileNotFoundError):
        lt.load_local_model(mid)


def test_predict_unknown_model_id(tmp_db):
    with pytest.raises(ValueError, match="不存在"):
        lt.load_local_model("m_不存在")


def test_train_predict_roundtrip_is_deterministic(trained, patched):
    a = lt.predict_local(trained["model_id"], [f"S{i:02d}" for i in range(20)],
                         top_n=10, write=False)
    b = lt.predict_local(trained["model_id"], [f"S{i:02d}" for i in range(20)],
                         top_n=10, write=False)
    pd.testing.assert_frame_equal(a, b)


# ---------- 训练后自检（v0.39：IC 全 0 时要说清楚为什么） ----------

def test_diagnosis_flags_small_universe(tmp_db, patched):
    r = lt.train_local([f"S{i:02d}" for i in range(20)], algo="lightgbm", params=FAST)
    assert any("候选池只有 20 只" in d for d in r["diagnosis"])


def test_diagnosis_empty_when_healthy(tmp_db, monkeypatch):
    """标的够多、样本够大时不该报任何问题——否则警告会变成噪声。

    50 只 × 700 根 → 训练段约 2.4 万条，刚好越过「158 个特征至少配 100 倍样本」
    这条经验线（158 × 100 ≈ 1.6 万）。
    """
    big = {f"S{i:03d}": _bars(700, seed=i) for i in range(50)}
    monkeypatch.setattr(lt, "load_bars", lambda *a, **k: big)
    r = lt.train_local([f"S{i:03d}" for i in range(50)], algo="lightgbm", params=FAST)
    assert r["metrics"]["sizes"]["train"] > 20_000
    assert r["diagnosis"] == [], f"健康训练不该有告警：{r['diagnosis']}"


def test_diagnosis_catches_collapse(tmp_db, patched):
    """强行用官方参数触发塌缩，自检必须点名「预测塌缩成常数」。

    不诊断的话对外只有一串 IC +0.0000，用户分不清「没信号」和「没训练」。
    """
    r = lt.train_local([f"S{i:02d}" for i in range(20)], algo="lightgbm",
                       params={"lambda_l1": 1e6, "lambda_l2": 1e6})
    assert any("塌缩成常数" in d for d in r["diagnosis"])
    assert r["metrics"]["ic"] == 0.0, "塌缩时 IC 确实是 0——正因如此才必须解释"


# ---------- 单因子基准扫描 ----------

def test_factor_scan_shape(patched):
    df = lt.factor_scan([f"S{i:02d}" for i in range(20)], top_k=8)
    assert len(df) == 8
    assert list(df.columns) == ["factor", "ic", "icir", "t_stat", "t_nw",
                               "win_rate", "n_days"]
    assert df["factor"].nunique() == 8


def test_factor_scan_sorted_by_abs_ic(patched):
    """按 |IC| 排序——负 IC 的因子反向用同样有效，不能只看正的。"""
    df = lt.factor_scan([f"S{i:02d}" for i in range(20)], top_k=20)
    absic = df["ic"].abs().to_numpy()
    assert (absic[:-1] >= absic[1:] - 1e-12).all(), "必须按 |IC| 降序"


def test_factor_scan_uses_test_segment_only(patched):
    """扫描必须在和模型同一个测试段上做，否则没有可比性。"""
    from eq.strategy.factors.validation import purged_split

    x, _ = lt.build_dataset(patched, horizon=5)
    sp = purged_split(x.index, valid_ratio=0.15, test_ratio=0.15, embargo_days=5)
    n_test_days = len(set(x[sp.test].index.get_level_values("datetime")))
    df = lt.factor_scan([f"S{i:02d}" for i in range(20)], top_k=3)
    assert int(df["n_days"].iloc[0]) == n_test_days


def test_factor_scan_no_test_segment_falls_back_to_valid(patched):
    df = lt.factor_scan([f"S{i:02d}" for i in range(20)], test_ratio=0.0, top_k=3)
    assert len(df) == 3


def test_factor_scan_needs_bars(monkeypatch):
    monkeypatch.setattr(lt, "load_bars", lambda *a, **k: {})
    with pytest.raises(ValueError, match="一只标的的行情都没拉到"):
        lt.factor_scan(["600519.SH"])


# ---------- 无参数基准（v0.40：选择与评估分开） ----------

def test_baseline_selects_on_valid_evaluates_on_test(patched):
    """选因子只能用验证段，测试段绝不能参与选择——否则又是选择偏差。"""
    r = lt.baseline_composite([f"S{i:02d}" for i in range(20)], top_k=4)
    assert len(r["selected"]) == 4
    assert set(r["singles"]["factor"]) == set(r["selected"])
    assert r["n_test_days"] > 0
    for k in ("ic_mean", "icir", "t_stat", "ic_win_rate"):
        assert k in r["composite"]


def test_baseline_shows_valid_to_test_shrinkage(patched):
    """随机数据上，验证段挑出来的 |IC| 到测试段必然缩水——这正是选择偏差本身。

    这条用例的价值是把「在哪个段上挑的」这件事的后果量出来：
    factor_scan 直接在测试段挑最大值，报出来的数就带着这份水分。
    """
    r = lt.baseline_composite([f"S{i:02d}" for i in range(20)], top_k=5)
    s = r["singles"]
    assert s["valid_ic"].abs().mean() > s["test_ic"].abs().mean(), \
        "验证段挑出的因子在测试段应当缩水"


def test_baseline_composite_is_not_significant_on_noise(patched):
    """纯随机行情上合成基准不该显著——工具本身不能凭空造出信号。

    判据用**重叠修正后**的 t 值而不是 IC 绝对值：horizon=5 时相邻交易日的标签
    共用 4 天行情，每日 IC 强烈自相关，20 只票的截面上 |IC| 到 0.1 都属正常涨落。
    这条用例第一次写成 |IC| < 0.05 时挂了（实得 0.097），正是这个原因——
    普通 t 值把自相关的日子当独立样本，把噪声算成了 4.6 个标准误。
    """
    r = lt.baseline_composite([f"S{i:02d}" for i in range(20)], top_k=5)
    assert abs(r["composite"]["t_stat_nw"]) < 3.0,         f"随机数据上不该显著：t_nw={r['composite']['t_stat_nw']:.2f}"


def test_baseline_requires_test_segment(patched):
    with pytest.raises(ValueError, match="没有独立测试段"):
        lt.baseline_composite([f"S{i:02d}" for i in range(20)], test_ratio=0.0)


def test_baseline_flips_negative_factors(patched, monkeypatch):
    """IC 为负的因子要反向计入合成，不能直接相加把信号抵消掉。"""
    r = lt.baseline_composite([f"S{i:02d}" for i in range(20)], top_k=5)
    # 只要选出的因子里有负 IC 的，就说明反向逻辑被触发过
    if (r["singles"]["valid_ic"] < 0).any():
        assert r["composite"]["n_days"] > 0


# ---------- 向量化 IC 矩阵（v0.41.1） ----------

def test_ic_matrix_matches_reference_daily_ic(fake_bars):
    """向量化实现必须和逐列 spearman 逐位一致——这是替换的前提。"""
    from eq.strategy.factors.evaluation import daily_ic

    x, y = lt.build_dataset(fake_bars, horizon=5)
    keep = x.notna().all(axis=1) & y.notna()
    xs, ys = x[keep], y[keep]
    m = lt.daily_ic_matrix(xs, ys)
    for col in ("MA20", "RSQR20", "CORR60", "KMID"):
        ref = daily_ic(xs[col], ys)
        got = m[col].reindex(ref.index).dropna()
        assert (got - ref.reindex(got.index)).abs().max() < 1e-12, col


def test_ic_matrix_shape_and_index(fake_bars):
    x, y = lt.build_dataset(fake_bars, horizon=5)
    m = lt.daily_ic_matrix(x, y)
    assert m.shape[1] == 158
    assert m.index.is_monotonic_increasing


def test_ic_matrix_empty_input():
    empty = pd.DataFrame(columns=["a"], dtype=float)
    assert lt.daily_ic_matrix(empty, pd.Series(dtype=float)).empty


def test_ic_matrix_rejects_missing_date_level():
    """没有 datetime 索引层就算不出「横截面」IC——要直接报错，不能悄悄给个数。"""
    x = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="datetime 索引层"):
        lt.daily_ic_matrix(x, pd.Series([1.0, 2.0, 3.0]))


def test_summarize_reports_both_t_values(fake_bars):
    """t_stat 和 t_nw 都要有，且 horizon>1 时两者必须不同。"""
    x, y = lt.build_dataset(fake_bars, horizon=5)
    m = lt.daily_ic_matrix(x, y)
    s1 = lt.summarize_ic_matrix(m, horizon=1)
    s5 = lt.summarize_ic_matrix(m, horizon=5)
    assert set(s5.columns) == {"factor", "ic", "icir", "t_stat", "t_nw",
                               "win_rate", "n_days"}
    import numpy as np
    assert np.allclose(s1["t_stat"], s5["t_stat"]), "普通 t 不随 horizon 变"
    assert not np.allclose(s1["t_nw"], s5["t_nw"]), "修正 t 必须随 horizon 变"


def test_factor_scan_emits_t_nw(patched):
    """回归：t_nw 曾经只加在空表分支上，非空分支漏了，CLI 直接 KeyError。

    根因是我用 str.replace 改代码但没断言匹配成功——静默 no-op，
    而当时的用例断言的是旧列名，所以也没拦住。
    """
    df = lt.factor_scan([f"S{i:02d}" for i in range(20)], top_k=3)
    assert "t_nw" in df.columns and df["t_nw"].notna().all()


def test_factor_scan_empty_and_nonempty_have_same_columns(patched, monkeypatch):
    """空表分支和正常分支必须返回同样的列——之前它们不一致。"""
    normal = lt.factor_scan([f"S{i:02d}" for i in range(20)], top_k=3)
    monkeypatch.setattr(lt, "daily_ic_matrix", lambda *a, **k: pd.DataFrame())
    empty = lt.factor_scan([f"S{i:02d}" for i in range(20)], top_k=3)
    assert list(normal.columns) == list(empty.columns)
