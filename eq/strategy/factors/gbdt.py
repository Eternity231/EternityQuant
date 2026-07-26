"""LightGBM 训练器（v0.38）—— 替掉 ``qlib.contrib.model.LGBModel``。

qlib 的 LGBModel 只是 LightGBM 的一层薄包装，但它把 ``qlib.data.dataset``
绑进了模型层：``fit(dataset)`` 要的是 qlib 的 DatasetH 对象，于是「训练一个
GBDT」这件事被迫依赖整套 qlib 数据栈。直接用 lightgbm 原生 API 之后：

- 模型层不再 import qlib，训练/预测都只吃 ``(DataFrame, Series)``
- 早停轮次、最佳迭代数这些以前藏在包装里的东西变成显式返回值
- **能测**——lightgbm 是本项目的直接依赖，造点合成数据就能跑通全流程；
  qlib 那条路要先有 .bin 数据才能验证一行代码

预测接口保持和自写 torch 模型一致（``predict(x)`` 吃 DataFrame），
所以 :func:`~eq.strategy.factors.ml_workflow._eval_segment` 那套评估、
:class:`~eq.strategy.factors.ml_workflow.SeedEnsemble` 集成都能直接复用。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["LGB_PARAMS", "GBDTModel", "train_gbdt"]

# qlib benchmarks 里 LightGBM+Alpha158 的官方参数。
# 两个 lambda 大得反直觉（205 / 580），但那是对的：Alpha158 有 158 个高度共线
# 的特征、标签信噪比极低，强 L1/L2 是防止它记住噪声的主要手段。
LGB_PARAMS: dict[str, Any] = {
    "objective": "mse",
    "learning_rate": 0.0421,
    "colsample_bytree": 0.8879,
    "subsample": 0.8789,
    "lambda_l1": 205.6999,
    "lambda_l2": 580.9769,
    "max_depth": 8,
    "num_leaves": 210,
    "feature_fraction": 0.8879,
    "bagging_fraction": 0.8789,
    "bagging_freq": 1,
    "min_data_in_leaf": 50,
    "num_threads": 20,
    "verbosity": -1,
}

# 训练轮数上限与早停耐心（不进 params，是 train() 的参数）
DEFAULT_ROUNDS = 1000
DEFAULT_EARLY_STOP = 50

# 上面那套官方参数是在 csi300 上调的：约 300 只 × 上千个交易日 ≈ 40 万样本。
# LightGBM 的 lambda_l1/l2 是**绝对量**惩罚（作用在叶子的梯度和上），
# 而梯度和随样本数线性增长——所以同一个 lambda 在小数据上等价于强了几十倍。
_REF_SAMPLES = 400_000


def scale_params_to_size(params: dict[str, Any], n_train: int) -> dict[str, Any]:
    """按训练样本量缩放正则强度和树容量。

    **这是一个实测出来的坑**：拿 5 只自选股训练（约 4000 个样本）时，
    ``lambda_l1=205.7`` 会让任何分裂的增益都盖不过惩罚项，LightGBM 直接退化成
    **一个常数叶子**——预测值全场相同，``best_iteration=1``。

    LightGBM 的 L1 是对叶子内**梯度和**做软阈值：``|Σg| ≤ lambda_l1`` 时叶子输出直接归零。
    所以塌缩与否同时取决于两件事——样本量（决定 Σg 能累到多大）和**标签尺度**。
    本项目默认对标签做截面 rank 归一化，把它压到 ±1.7，梯度和比原始收益率标签
    小得多，更容易被削平（实测：同样 336 个样本，原始收益率标签不塌缩、rank 标签塌缩）。
    截面 IC 于是恒等于 0（同一天所有票分数一样，排不出序），
    输出一串 ``IC +0.0000 / ICIR 0.000 / 胜率 0%``，看起来像"这批票没信号"，
    实际是"模型根本没长出来"。实测同一份数据换轻正则后 valid IC 有 0.127。

    缩放规则：

    - ``lambda_l1/lambda_l2`` 按 ``n_train / 40 万`` 线性缩放（有下限，别缩到 0）
    - ``num_leaves`` 不超过 ``n_train / 100``——4000 个样本配 210 片叶子，
      平均每片不到 20 个样本，纯粹在记噪声
    - ``min_data_in_leaf`` 不超过 ``n_train / 50``，否则小数据上没有一个分裂合法

    显式传了这些键就照传的来，不缩放。
    """
    out = dict(params)
    if n_train <= 0 or n_train >= _REF_SAMPLES:
        return out
    ratio = n_train / _REF_SAMPLES
    for key, floor in (("lambda_l1", 0.1), ("lambda_l2", 0.1)):
        if key in out:
            out[key] = max(floor, float(out[key]) * ratio)
    if "num_leaves" in out:
        out["num_leaves"] = int(max(7, min(int(out["num_leaves"]), n_train // 100)))
    if "min_data_in_leaf" in out:
        out["min_data_in_leaf"] = int(max(5, min(int(out["min_data_in_leaf"]),
                                                 n_train // 50)))
    return out


class GBDTModel:
    """训练好的 LightGBM，接口对齐项目里的自写 torch 模型。

    ``predict`` 只接一个位置参数，:func:`_eval_segment` 先试 qlib 签名
    ``predict(dataset, segment=)`` 会抛 TypeError 并退回来——和
    :class:`SeedEnsemble` 走的是同一条兼容路径。
    """

    def __init__(self, booster, feature_names: list[str],
                 best_iteration: int, best_score: float):
        self.booster = booster
        self.feature_names = list(feature_names)
        self.best_iteration = int(best_iteration)
        self.best_score = float(best_score)
        self.best_step = int(best_iteration)     # 和 torch 模型的字段名对齐
        # 预测是否塌缩成常数（由 train_gbdt 回填），供上层给出人话诊断
        self.collapsed = False
        self.effective_params: dict[str, Any] = {}

    def predict(self, x):
        if isinstance(x, pd.DataFrame):
            missing = [c for c in self.feature_names if c not in x.columns]
            if missing:
                raise ValueError(f"特征列缺失 {missing[:5]}（共 {len(missing)} 列）")
            arr = x[self.feature_names].to_numpy(dtype="float64")
        else:
            arr = np.asarray(x, dtype="float64")
        return self.booster.predict(arr, num_iteration=self.best_iteration or None)

    def feature_importance(self, top: int = 20) -> pd.Series:
        """按增益排序的特征重要性。看看模型到底在用哪些因子。"""
        gain = self.booster.feature_importance(importance_type="gain")
        return (pd.Series(gain, index=self.feature_names)
                .sort_values(ascending=False).head(top))


def train_gbdt(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame | None = None,
    y_valid: pd.Series | None = None,
    params: dict[str, Any] | None = None,
    num_boost_round: int = DEFAULT_ROUNDS,
    early_stopping_rounds: int = DEFAULT_EARLY_STOP,
    device: str = "cpu",
    seed: int = 42,
    verbose: bool = False,
    auto_scale: bool = True,
) -> GBDTModel:
    """训练一个 LightGBM。有验证集就按验证集早停。

    ``device``：``cpu`` | ``gpu``（LightGBM 的 OpenCL 后端）。传 ``cuda``
    会被当成 ``gpu``——LightGBM 没有单独的 cuda 取值，而项目里 torch 那边用
    ``cuda``，两边混用过。
    """
    import lightgbm as lgb

    # 先按样本量缩放默认正则，再让用户显式传的参数覆盖——
    # 顺序反过来的话用户传的值会被缩放掉
    p = scale_params_to_size(LGB_PARAMS, len(y_train)) if auto_scale else dict(LGB_PARAMS)
    p.update(params or {})
    p["seed"] = seed
    dev = {"cuda": "gpu"}.get(str(device).lower(), str(device).lower())
    if dev in ("gpu", "cpu"):
        p["device"] = dev

    feats = list(x_train.columns)
    dtrain = lgb.Dataset(x_train.to_numpy(dtype="float64"),
                         label=np.asarray(y_train, dtype="float64"),
                         feature_name=feats, free_raw_data=False)

    valid_sets, callbacks = [], []
    has_valid = (x_valid is not None and y_valid is not None
                 and len(x_valid) > 0 and len(y_valid) > 0)
    if has_valid:
        dvalid = lgb.Dataset(x_valid[feats].to_numpy(dtype="float64"),
                             label=np.asarray(y_valid, dtype="float64"),
                             reference=dtrain, free_raw_data=False)
        valid_sets = [dvalid]
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=verbose))
    else:
        # 没有验证集就没法早停，跑满轮数几乎必然过拟合——明确警告，不静默
        logger.warning("无验证集，LightGBM 将跑满 %d 轮且不早停，结果大概率过拟合",
                       num_boost_round)
    if verbose:
        callbacks.append(lgb.log_evaluation(period=50))

    booster = lgb.train(p, dtrain, num_boost_round=num_boost_round,
                        valid_sets=valid_sets, callbacks=callbacks)

    best_iter = booster.best_iteration or num_boost_round
    best_score = 0.0
    collapsed = False
    if has_valid:
        # 用**每日横截面 Rank IC** 记分，和早停口径、最终验收保持同一把尺
        # （早停本身只能用 LightGBM 的 mse，那是它内部的事）
        from eq.strategy.factors.evaluation import daily_ic

        pred = booster.predict(x_valid[feats].to_numpy(dtype="float64"),
                               num_iteration=best_iter)
        # 预测塌缩成常数 = 模型没长出来。这种情况下截面 IC 恒为 0
        # （同一天所有票分数相同，排不出序），报出来是一串漂亮的 0，
        # 看着像"没信号"其实是"没训练"——必须当场喊出来。
        if float(np.std(pred)) < 1e-12:
            collapsed = True
            logger.error(
                "LightGBM 预测塌缩成常数（best_iteration=%d）——模型没有学到任何分裂。"
                "常见原因：正则太强 / 样本太少 / 特征全是常数。"
                "当前 lambda_l1=%s lambda_l2=%s num_leaves=%s min_data_in_leaf=%s，"
                "训练样本 %d 条", best_iter, p.get("lambda_l1"), p.get("lambda_l2"),
                p.get("num_leaves"), p.get("min_data_in_leaf"), len(y_train))
        ics = daily_ic(pd.Series(pred, index=y_valid.index), y_valid)
        if len(ics):
            m = float(ics.mean())
            best_score = m if m == m else 0.0
    model = GBDTModel(booster, feats, best_iter, best_score)
    model.collapsed = collapsed
    model.effective_params = p
    return model
