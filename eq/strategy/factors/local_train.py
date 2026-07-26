"""无 qlib 训练链路（v0.39）—— 从 OHLCV 到模型，全程不碰 qlib。

链路：

    项目自己的行情缓存 → alpha.alpha158（158 特征）
      → validation.purged_split（按交易日切 + embargo）
      → preprocess.Pipeline（只在训练段 fit）
      → gbdt / MLPAlphaNet / RecurrentAlphaNet
      → evaluation.evaluate（逐日截面 Rank IC）
      → ml_models 表

和 :func:`ml_workflow.train` 的区别只有一处：**数据从哪来**。
那条路要 qlib 的 .bin + 表达式引擎 + 三个 monkey patch；这条路直接吃
:func:`eq.data.market.get_recent_bars`，也就是 ``eq data a`` 下下来的那份缓存。

其余环节（切分、预处理、模型、评估、集成、注册）两条路完全共用，
所以两边的成绩可以直接比——比的是特征实现，不是别的东西。
"""

from __future__ import annotations

import datetime as dt
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd

from eq.strategy.factors import alpha as alpha_mod
from eq.strategy.factors import preprocess as pp
from eq.strategy.factors.ml import register_model
from eq.strategy.factors.validation import purged_split, set_seed

logger = logging.getLogger(__name__)

__all__ = ["load_bars", "build_dataset", "train_local"]


def load_bars(symbols: list[str], days: int = 1200, workers: int = 8,
              use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """并发拉一篮子标的的日线。取不到的标的跳过并记日志，不中断整体。"""
    from eq.data.market import get_recent_bars

    def _one(sym: str):
        try:
            return sym, get_recent_bars(sym, days=days, use_cache=use_cache)
        except Exception as e:
            logger.warning("行情拉取失败 %s：%s", sym, e)
            return sym, None

    out: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(symbols) or 1))) as pool:
        for sym, df in pool.map(_one, symbols):
            if df is not None and len(df):
                out[sym] = df
    return out


def build_dataset(bars: dict[str, pd.DataFrame], horizon: int = 5,
                  label_norm: str = "rank") -> tuple[pd.DataFrame, pd.Series]:
    """行情 → (特征面板, 标签)。已丢掉标签缺失的样本。

    标签在**切分之前**做截面归一化：rank 是当日截面内的相对次序，
    不涉及跨期信息，所以不会造成泄漏。特征的归一化则必须放到切分之后
    （统计量只能用训练段拟合），那一步在 :func:`train_local` 里。
    """
    x = alpha_mod.alpha158(bars)
    y = alpha_mod.forward_return(bars, horizon=horizon)
    if x.empty or y.empty:
        raise ValueError("特征或标签为空——检查行情是否拉到、根数是否够长")
    x, y = x.align(y, join="inner", axis=0)
    x, y = pp.dropna_label(x, y)
    if len(y) == 0:
        raise ValueError("丢掉标签缺失后没有样本了")
    return x, pp.normalize_label(y, label_norm)


def train_local(
    symbols: list[str],
    *,
    algo: str = "lightgbm",
    horizon: int = 5,
    days: int = 1200,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    embargo_days: int | None = None,
    label_norm: str = "rank",
    device: str = "cpu",
    seed: int = 42,
    n_seeds: int = 1,
    name: str | None = None,
    universe_label: str = "local",
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """跑一次完整训练，全程不依赖 qlib。

    Args:
        symbols: 标的列表（项目标准符号，如 ``600519.SH``）
        algo: ``lightgbm`` | ``mlp`` | ``gru`` | ``lstm``
        embargo_days: 段间 purge 的交易日数，缺省取 ``horizon``
            ——标签用到了 T+h 的价格，不 purge 就是泄漏
        n_seeds: >1 时训练多个种子并集成（见 :class:`SeedEnsemble`）

    Returns:
        ``{"model_id", "metrics", "model_path", "n_symbols", "n_samples"}``
    """
    set_seed(seed)
    embargo = horizon if embargo_days is None else int(embargo_days)

    bars = load_bars(symbols, days=days)
    if not bars:
        raise ValueError("一只标的的行情都没拉到")
    x, y = build_dataset(bars, horizon=horizon, label_norm=label_norm)

    split = purged_split(x.index, valid_ratio=valid_ratio, test_ratio=test_ratio,
                         embargo_days=embargo, with_test=test_ratio > 0)
    sizes = split.sizes()
    logger.info("样本 %s  特征 %d 维  标的 %d 只", sizes, x.shape[1], len(bars))
    if sizes["train"] < 100 or sizes["valid"] < 20:
        raise ValueError(f"切分后样本太少：{sizes}（拉长 days 或多加几只标的）")

    # 特征归一化的统计量**只在训练段拟合**
    pipe = pp.default_pipeline().fit(x[split.train])
    x_train, y_train = pipe.transform(x[split.train]), y[split.train]
    x_valid, y_valid = pipe.transform(x[split.valid]), y[split.valid]

    model, notes = _fit(algo, x_train, y_train, x_valid, y_valid,
                        device=device, seed=seed, n_seeds=n_seeds, params=params)

    valid_ic = float(getattr(model, "best_score", 0.0))
    test_report = None
    if split.test is not None and split.test.sum() > 0:
        from eq.strategy.factors.evaluation import evaluate

        x_test, y_test = pipe.transform(x[split.test]), y[split.test]
        pred = pd.Series(model.predict(x_test), index=x_test.index)
        test_report = evaluate(pred, y_test)
    ic = test_report["ic_mean"] if test_report else valid_ic

    import pickle as _pkl

    from eq.strategy.factors.ml_workflow import _ensure_dir

    suffix = f"_x{n_seeds}" if n_seeds > 1 else ""
    model_path = _ensure_dir() / f"local_{algo}_{horizon}d{suffix}.pkl"
    with open(model_path, "wb") as f:
        _pkl.dump({"model": model, "pipeline": pipe,
                   "features": list(x.columns), "horizon": horizon}, f)

    b = split.bounds
    model_id = register_model(
        name=name or f"local_{algo}_h{horizon}_{dt.date.today().strftime('%Y%m%d')}",
        universe=universe_label,
        features=["alpha158_local"],
        algo=algo,
        horizon=horizon,
        train_period=f"{b.get('train', ('', ''))[0]}~{b.get('train', ('', ''))[1]}",
        valid_period=f"{b.get('valid', ('', ''))[0]}~{b.get('valid', ('', ''))[1]}",
        metrics={"ic": ic, "valid_ic": valid_ic, "algo": algo, "horizon": horizon,
                 "seed": seed, "n_seeds": n_seeds, "device": device,
                 "embargo_days": embargo, "label_norm": label_norm,
                 "n_symbols": len(bars), "feature_set": "alpha158_local",
                 **{f"test_{k}": v for k, v in (test_report or {}).items()
                    if k != "ic_series" and not isinstance(v, list)}},
        model_path=str(model_path),
        notes=notes + "｜本地特征，无 qlib",
    )
    return {"model_id": model_id, "model_path": str(model_path),
            "n_symbols": len(bars), "n_samples": int(len(y)),
            "metrics": {"ic": ic, "valid_ic": valid_ic, "test": test_report,
                        "sizes": sizes}}


def _fit(algo, x_train, y_train, x_valid, y_valid, *, device, seed, n_seeds, params):
    """按 algo 训练（必要时多种子集成），返回 ``(model, notes)``。"""
    from eq.strategy.factors.ml_workflow import (
        MLPAlphaNet, RecurrentAlphaNet, SeedEnsemble,
    )

    n_feat = x_train.shape[1]

    if algo == "lightgbm":
        from eq.strategy.factors.gbdt import train_gbdt

        def _make(s):
            return train_gbdt(x_train, y_train, x_valid, y_valid,
                              params=params, device=device, seed=s)
        notes = f"本地 Alpha158 + LightGBM（{device}）"
    elif algo == "mlp":
        def _make(s):
            m = MLPAlphaNet(input_dim=n_feat, hidden=(512, 256, 128),
                            device=device, seed=s)
            m.fit(x_train, y_train, x_valid, y_valid, early_stop=30)
            return m
        notes = f"本地 Alpha158 + MLPAlphaNet（{device}）"
    elif algo in ("gru", "lstm"):
        def _make(s):
            m = RecurrentAlphaNet(input_dim=n_feat, cell_type=algo,
                                  device=device, seed=s)
            m.fit(x_train, y_train, x_valid, y_valid, early_stop=20)
            return m
        notes = f"本地 Alpha158 + {algo.upper()}（{device}）"
    else:
        raise ValueError(f"未知 algo {algo}（可选 lightgbm/mlp/gru/lstm）")

    n = max(1, int(n_seeds))
    members = []
    for i in range(n):
        if n > 1:
            logger.info("集成 %d/%d（seed=%d）", i + 1, n, seed + i)
        members.append(_make(seed + i))
    if n == 1:
        return members[0], notes
    return SeedEnsemble(members, [seed + i for i in range(n)]), notes + f"｜{n} 种子集成"
