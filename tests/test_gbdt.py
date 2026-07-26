"""原生 LightGBM 训练器（v0.38）。

替掉 qlib.contrib.model.LGBModel 的直接收益：这些用例**能跑**。
走 qlib 那条路要先有一整套 .bin 数据才能验证一行代码。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from eq.strategy.factors.gbdt import (  # noqa: E402
    LGB_PARAMS, GBDTModel, train_gbdt,
)

FAST = {"num_leaves": 7, "min_data_in_leaf": 5, "lambda_l1": 0.0,
        "lambda_l2": 0.0, "learning_rate": 0.1}


def _panel(n_days=40, n_stocks=25, n_feat=6, seed=0, signal=1.0):
    """造一份「f0 有预测力、其余是噪声」的面板。"""
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range("2024-01-01", periods=n_days),
         [f"S{i:02d}" for i in range(n_stocks)]],
        names=["datetime", "instrument"])
    x = pd.DataFrame(rng.normal(size=(len(idx), n_feat)),
                     index=idx, columns=[f"f{i}" for i in range(n_feat)])
    y = pd.Series(signal * x["f0"] + rng.normal(scale=0.5, size=len(idx)), index=idx)
    return x, y


def _split(x, y, frac=0.7):
    days = x.index.get_level_values("datetime").unique()
    cut = days[int(len(days) * frac)]
    m = x.index.get_level_values("datetime") < cut
    return x[m], y[m], x[~m], y[~m]


# ---------- 基本训练 ----------

def test_learns_the_real_signal():
    xt, yt, xv, yv = _split(*_panel())
    m = train_gbdt(xt, yt, xv, yv, params=FAST, num_boost_round=60,
                   early_stopping_rounds=20)
    assert m.best_score > 0.3, f"信号明确时 valid Rank IC 应该不低：{m.best_score}"


def test_predict_accepts_dataframe_and_keeps_length():
    xt, yt, xv, yv = _split(*_panel())
    m = train_gbdt(xt, yt, xv, yv, params=FAST, num_boost_round=30)
    p = m.predict(xv)
    assert len(p) == len(xv) and np.isfinite(p).all()


def test_predict_reorders_columns():
    """列顺序被打乱也要给出同样的结果——按列名取，不按位置。"""
    xt, yt, xv, yv = _split(*_panel())
    m = train_gbdt(xt, yt, xv, yv, params=FAST, num_boost_round=30)
    shuffled = xv[list(reversed(xv.columns))]
    np.testing.assert_allclose(m.predict(xv), m.predict(shuffled))


def test_predict_rejects_missing_columns():
    xt, yt, xv, yv = _split(*_panel())
    m = train_gbdt(xt, yt, xv, yv, params=FAST, num_boost_round=20)
    with pytest.raises(ValueError, match="特征列缺失"):
        m.predict(xv.drop(columns=["f0"]))


def test_early_stopping_actually_stops():
    """噪声数据上应该早早停下，而不是跑满 500 轮。"""
    x, y = _panel(signal=0.0, seed=3)          # 纯噪声，学不到东西
    xt, yt, xv, yv = _split(x, y)
    m = train_gbdt(xt, yt, xv, yv, params=FAST, num_boost_round=500,
                   early_stopping_rounds=10)
    assert m.best_iteration < 500, f"纯噪声不该跑满：{m.best_iteration}"


def test_without_valid_runs_full_rounds_and_warns(caplog):
    xt, yt, _, _ = _split(*_panel())
    with caplog.at_level("WARNING"):
        m = train_gbdt(xt, yt, params=FAST, num_boost_round=25)
    assert m.best_iteration == 25
    assert any("不早停" in r.message for r in caplog.records), "没验证集必须明确警告"
    assert m.best_score == 0.0


# ---------- 可复现与可存盘 ----------

def test_same_seed_same_result():
    xt, yt, xv, yv = _split(*_panel())
    a = train_gbdt(xt, yt, xv, yv, params=FAST, num_boost_round=40, seed=7)
    b = train_gbdt(xt, yt, xv, yv, params=FAST, num_boost_round=40, seed=7)
    np.testing.assert_allclose(a.predict(xv), b.predict(xv))


def test_model_is_picklable():
    import pickle

    xt, yt, xv, yv = _split(*_panel())
    m = train_gbdt(xt, yt, xv, yv, params=FAST, num_boost_round=20)
    back = pickle.loads(pickle.dumps(m))
    assert isinstance(back, GBDTModel)
    np.testing.assert_allclose(back.predict(xv), m.predict(xv))


# ---------- 和评估/集成层的兼容 ----------

def test_predict_signature_falls_back_from_qlib():
    """_eval_segment 先试 qlib 的 predict(dataset, segment=)，必须抛 TypeError。"""
    xt, yt, xv, yv = _split(*_panel())
    m = train_gbdt(xt, yt, xv, yv, params=FAST, num_boost_round=20)
    with pytest.raises(TypeError):
        m.predict(xv, segment="test")


def test_works_inside_seed_ensemble():
    from eq.strategy.factors.ml_workflow import SeedEnsemble

    xt, yt, xv, yv = _split(*_panel())
    members = [train_gbdt(xt, yt, xv, yv, params=FAST, num_boost_round=30, seed=s)
               for s in (1, 2, 3)]
    ens = SeedEnsemble(members, seeds=[1, 2, 3])
    out = ens.predict(xv)
    assert len(out) == len(xv) and np.isfinite(out).all()


def test_feature_importance_finds_the_real_feature():
    """f0 是唯一有信号的特征，重要性该排第一——顺带验证模型没在学噪声。"""
    xt, yt, xv, yv = _split(*_panel(signal=3.0))
    m = train_gbdt(xt, yt, xv, yv, params=FAST, num_boost_round=80)
    assert m.feature_importance(top=3).index[0] == "f0"


# ---------- 参数与设备 ----------

def test_official_params_are_heavily_regularized():
    """qlib benchmarks 的官方参数：两个 lambda 大得反直觉，但那是对的。"""
    assert LGB_PARAMS["lambda_l1"] > 100 and LGB_PARAMS["lambda_l2"] > 100


def test_cuda_device_maps_to_gpu(monkeypatch):
    """torch 那边用 cuda，LightGBM 只认 gpu——两边混用过，必须映射。"""
    import lightgbm as lgb

    seen: dict = {}
    real = lgb.train

    def _spy(p, *a, **k):
        seen.update(p)
        return real({**p, "device": "cpu"}, *a, **k)

    monkeypatch.setattr(lgb, "train", _spy)
    xt, yt, xv, yv = _split(*_panel())
    train_gbdt(xt, yt, xv, yv, params=FAST, num_boost_round=5, device="cuda")
    assert seen.get("device") == "gpu"


def test_user_params_override_defaults():
    xt, yt, xv, yv = _split(*_panel())
    m = train_gbdt(xt, yt, xv, yv, params={**FAST, "num_leaves": 3},
                   num_boost_round=10)
    assert m.booster.params.get("num_leaves") == 3
