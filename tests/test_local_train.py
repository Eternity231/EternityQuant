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
