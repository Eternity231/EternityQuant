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


# ---------- 小样本上的正则塌缩（v0.39 实测到的真 BUG） ----------

def _ranked_panel(n_days, n_stocks, seed=0, signal=2.0):
    """标签做截面 rank 归一化——train_local 的默认口径。

    这一步很关键：rank 归一化把标签压到 ±1.7，梯度和随之变小，
    更容易被 lambda_l1 的阈值整个削平。同样的样本量，
    原始收益率标签不塌缩、rank 标签就塌缩（实测）。
    """
    from eq.strategy.factors.preprocess import cs_rank_norm

    x, y = _panel(n_days=n_days, n_stocks=n_stocks, seed=seed, signal=signal)
    return x, cs_rank_norm(y)


def test_official_params_collapse_on_small_data():
    """反证：不缩放的话，官方参数会把模型压成一个常数叶子。

    这是用户实跑 5 只自选股时真实发生的——输出一串 IC +0.0000 / 胜率 0%，
    看着像「这批票没信号」，实际是「模型根本没长出来」。
    """
    from eq.strategy.factors.gbdt import train_gbdt as tg

    xt, yt, xv, yv = _split(*_ranked_panel(60, 8))
    m = tg(xt, yt, xv, yv, num_boost_round=100, auto_scale=False)
    assert m.collapsed, "构造前提：官方参数在这个规模上应当塌缩"
    assert len(np.unique(np.round(m.predict(xv), 12))) == 1
    assert m.best_iteration == 1, "一个分裂都没成立"


def test_auto_scale_prevents_collapse():
    """开启缩放（默认）后同样的数据能学出东西。"""
    xt, yt, xv, yv = _split(*_ranked_panel(60, 8))
    m = train_gbdt(xt, yt, xv, yv, num_boost_round=100)
    assert not m.collapsed
    assert len(np.unique(np.round(m.predict(xv), 12))) > 1
    assert m.best_score > 0.1, f"信号明确时该学得到：{m.best_score}"


def test_scale_shrinks_regularization_proportionally():
    from eq.strategy.factors.gbdt import scale_params_to_size

    small = scale_params_to_size(LGB_PARAMS, 4_000)
    assert small["lambda_l1"] < LGB_PARAMS["lambda_l1"] / 50
    assert small["num_leaves"] <= 4_000 // 100
    assert small["min_data_in_leaf"] <= 4_000 // 50


def test_scale_is_identity_on_large_data():
    """样本量达到参考规模时不缩放——官方参数本来就是给这个量级调的。"""
    from eq.strategy.factors.gbdt import scale_params_to_size

    assert scale_params_to_size(LGB_PARAMS, 500_000) == LGB_PARAMS


def test_scale_has_a_floor():
    """再小的数据也不能把正则缩到 0，否则纯过拟合。"""
    from eq.strategy.factors.gbdt import scale_params_to_size

    tiny = scale_params_to_size(LGB_PARAMS, 10)
    assert tiny["lambda_l1"] >= 0.1 and tiny["lambda_l2"] >= 0.1
    assert tiny["num_leaves"] >= 7 and tiny["min_data_in_leaf"] >= 5


def test_explicit_params_are_not_scaled():
    """用户显式给的值必须原样生效，不能被缩放偷偷改掉。"""
    xt, yt, xv, yv = _split(*_panel(n_days=60, n_stocks=8))
    m = train_gbdt(xt, yt, xv, yv, params={"lambda_l1": 99.0, "num_leaves": 3},
                   num_boost_round=10)
    assert m.effective_params["lambda_l1"] == 99.0
    assert m.effective_params["num_leaves"] == 3


# ---------- 早停口径（v0.40） ----------

def test_ic_early_stop_uses_custom_metric():
    """开启后内置 metric 关掉，改用 Rank IC 早停——选和考同一把尺。"""
    xt, yt, xv, yv = _split(*_ranked_panel(80, 20))
    m = train_gbdt(xt, yt, xv, yv, num_boost_round=200, ic_early_stop=True)
    assert m.effective_params.get("metric") == "None"
    assert m.best_iteration >= 1


def test_mse_early_stop_keeps_builtin_metric():
    xt, yt, xv, yv = _split(*_ranked_panel(80, 20))
    m = train_gbdt(xt, yt, xv, yv, num_boost_round=200, ic_early_stop=False)
    assert m.effective_params.get("metric") != "None"


def test_ic_feval_returns_higher_is_better():
    """feval 的第三个返回值必须是 True，否则 LightGBM 会往 IC 变小的方向早停。"""
    from eq.strategy.factors.gbdt import _make_ic_feval

    _, y = _ranked_panel(20, 10)
    name, value, higher = _make_ic_feval(y)(np.asarray(y), None)
    assert name == "rank_ic" and higher is True
    assert value > 0.9, "拿标签自己当预测，IC 应该接近 1"


def test_ic_feval_handles_constant_prediction():
    from eq.strategy.factors.gbdt import _make_ic_feval

    _, y = _ranked_panel(20, 10)
    _, value, _ = _make_ic_feval(y)(np.zeros(len(y)), None)
    assert value == 0.0, "常数预测的 IC 是 0，不能是 NaN"
