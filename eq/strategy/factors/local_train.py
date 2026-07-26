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

import numpy as np
import pandas as pd

from eq.strategy.factors import alpha as alpha_mod
from eq.strategy.factors import preprocess as pp
from eq.strategy.factors.ml import register_model
from eq.strategy.factors.validation import purged_split, set_seed

logger = logging.getLogger(__name__)

# 截面选股的候选池下限（低于此值只警告不拦截）
_MIN_UNIVERSE = 30

__all__ = ["load_bars", "build_dataset", "train_local",
           "load_local_model", "predict_local", "factor_scan"]


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
    # 截面选股是「今天这批票里挑哪只」，候选池太小的话每天只在几只之间排序，
    # IC 的方差极大、几乎没有统计意义。不拦（用户可能就想试试），但要说清楚。
    if len(bars) < _MIN_UNIVERSE:
        logger.warning(
            "候选池只有 %d 只（建议 ≥%d）。截面模型每天只能在这几只之间排序，"
            "IC 噪声极大；试试 --from A --top 300 或多加些自选股",
            len(bars), _MIN_UNIVERSE)
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
    # 预测塌缩成常数时，截面 IC 恒等于 0（同一天所有票分数相同，排不出序）。
    # 不诊断的话对外就是一串漂亮的 "IC +0.0000 / 胜率 0%"，
    # 看着像「这批票没信号」，实际是「模型根本没长出来」——两者的处置完全不同。
    diagnosis = _diagnose(model, sizes["train"], len(bars))
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
            "diagnosis": diagnosis,
            "metrics": {"ic": ic, "valid_ic": valid_ic, "test": test_report,
                        "sizes": sizes}}


def _diagnose(model, n_train: int, n_symbols: int) -> list[str]:
    """训练后自检，返回给人看的问题清单（没问题就是空列表）。

    专治「指标全 0 但不知道为什么」：IC=0 有两种截然不同的原因——
    模型没长出来（要调参）和这批票真没信号（要换票），必须分清。
    """
    out: list[str] = []
    members = getattr(model, "models", [model])
    if any(getattr(m, "collapsed", False) for m in members):
        eff = next((getattr(m, "effective_params", {}) for m in members
                    if getattr(m, "collapsed", False)), {})
        out.append(
            f"模型预测塌缩成常数——没有学到任何分裂，IC 恒为 0。"
            f"当前 lambda_l1={eff.get('lambda_l1')} lambda_l2={eff.get('lambda_l2')}"
            f" num_leaves={eff.get('num_leaves')}，训练样本 {n_train} 条。"
            f"试试更小的正则：--params 或直接扩大候选池")
    if n_symbols < _MIN_UNIVERSE:
        out.append(f"候选池只有 {n_symbols} 只（建议 ≥{_MIN_UNIVERSE}）："
                   f"截面模型每天只在这几只之间排序，IC 噪声极大")
    if n_train < 20_000:
        out.append(f"训练样本仅 {n_train:,} 条：Alpha158 有 158 维特征，"
                   f"样本太少容易记住噪声。加标的或拉长 --days")
    return out


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


# ======================================================================
# 推理
# ======================================================================

def load_local_model(model_id: str) -> dict[str, Any]:
    """按 model_id 从 ml_models 表读出本地模型（含预处理管线）。

    存的是 ``{"model", "pipeline", "features", "horizon"}`` 而不是裸模型——
    **管线必须和模型一起走**。推理时重新 fit 一个管线，得到的归一化统计量
    来自推理数据而不是训练段，那就是 v0.37 修掉的 train/serve skew 又回来了。
    """
    import pickle
    from pathlib import Path

    from eq.db import execute

    rows = execute("SELECT model_path, notes FROM ml_models WHERE id = ?", (model_id,))
    if not rows:
        raise ValueError(f"模型 {model_id} 不存在")
    path = Path(rows[0]["model_path"])
    if not path.exists():
        raise FileNotFoundError(f"模型文件不见了：{path}")
    blob = pickle.loads(path.read_bytes())
    if not isinstance(blob, dict) or "pipeline" not in blob:
        raise ValueError(
            f"{model_id} 不是本地模型（没有预处理管线）。"
            "走 qlib 训练的模型请用 `eq ml predict-batch`")
    return blob


def predict_local(
    model_id: str,
    symbols: list[str],
    *,
    top_n: int = 20,
    days: int = 400,
    predict_date: str | None = None,
    write: bool = True,
) -> pd.DataFrame:
    """用本地模型给一批标的打分，默认写入 ``ml_predictions`` 表。

    只对**最后一个共同交易日**的截面打分：选股是个截面排序问题，
    跨日混在一起排没有意义（不同日期的分数不可比）。

    Args:
        predict_date: 指定打分日期（``YYYY-MM-DD``）。缺省用数据里的最后一天。
        write: False 时只返回结果不落库，方便先看看再决定。

    Returns:
        ``symbol`` / ``score`` 两列，按分数降序，最多 ``top_n`` 行。
    """
    blob = load_local_model(model_id)
    model, pipe, feats = blob["model"], blob["pipeline"], blob["features"]

    bars = load_bars(symbols, days=days)
    if not bars:
        raise ValueError("一只标的的行情都没拉到")
    x = alpha_mod.alpha158(bars)
    if x.empty:
        raise ValueError("特征为空——多半是行情根数不够（至少要 70 根）")

    dates = x.index.get_level_values("datetime")
    target = pd.Timestamp(predict_date) if predict_date else dates.max()
    cross = x[dates == target]
    if cross.empty:
        avail = sorted(set(dates))[-3:]
        raise ValueError(f"{target.date()} 没有数据，最近可用：{[str(d.date()) for d in avail]}")

    missing = [c for c in feats if c not in cross.columns]
    if missing:
        raise ValueError(f"特征对不上，缺 {missing[:5]}（共 {len(missing)}）——"
                         "模型和当前代码的特征集不一致，需要重训")
    # 用**训练时**拟合好的管线，不重新 fit
    scored = pd.DataFrame({
        "symbol": cross.index.get_level_values("instrument"),
        "score": np.asarray(model.predict(pipe.transform(cross[feats]))).reshape(-1),
    }).sort_values("score", ascending=False).head(top_n).reset_index(drop=True)

    if write and len(scored):
        from eq.db import execute_write

        for _, row in scored.iterrows():
            execute_write(
                "INSERT INTO ml_predictions (model_id, symbol, date, score) "
                "VALUES (?, ?, ?, ?)",
                (model_id, row["symbol"], target.date().isoformat(), float(row["score"])),
            )
    scored.attrs["date"] = target.date().isoformat()
    return scored


# ======================================================================
# 单因子基准扫描
# ======================================================================

def factor_scan(
    symbols: list[str],
    *,
    horizon: int = 5,
    days: int = 1200,
    test_ratio: float = 0.15,
    valid_ratio: float = 0.15,
    embargo_days: int | None = None,
    top_k: int = 15,
) -> pd.DataFrame:
    """在**和模型完全相同的测试段**上，逐个评估 158 个单因子。

    这是训练结果唯一有意义的参照物。模型 test IC 报出来 +0.011 时，
    单看这个数没法判断——它可能意味着两件完全不同的事：

    - 最好的单因子也只有 0.01 → 这段行情/这批票就是难，模型没做错什么
    - 有单因子能到 0.05 → 158 维模型跑输一个不用训练的公式，管线有问题

    不训练、不调参，纯算截面 Rank IC，所以很快。

    **多重检验警告**：这是从 158 个因子里挑最大值，不是单次检验。零假设下
    扫 158 个因子，最大 |t| 本来就期望在 3 附近——在纯随机数据上实测能挑出
    ``t=4.85`` 的"因子"。Bonferroni 校正（α=0.05/158）后，**排第一的那个**
    要 ``|t| > 3.6`` 才谈得上显著。这个函数的用途是给模型成绩当**参照物**，
    不是用来挖因子的。

    Returns:
        按 |IC| 降序的前 ``top_k`` 行，列：``factor / ic / icir / t_stat /
        win_rate / n_days``。IC 为负的因子反向用同样有效，所以排序看绝对值，
        ``ic`` 列保留原始符号。
    """
    from eq.strategy.factors.evaluation import evaluate

    embargo = horizon if embargo_days is None else int(embargo_days)
    bars = load_bars(symbols, days=days)
    if not bars:
        raise ValueError("一只标的的行情都没拉到")
    x, y = build_dataset(bars, horizon=horizon)
    split = purged_split(x.index, valid_ratio=valid_ratio, test_ratio=test_ratio,
                         embargo_days=embargo, with_test=test_ratio > 0)
    mask = split.test if split.test is not None else split.valid
    x_t, y_t = x[mask], y[mask]
    logger.info("单因子扫描：%d 个因子 × %d 条测试样本", x_t.shape[1], len(y_t))

    rows = []
    for col in x_t.columns:
        f = x_t[col]
        if f.notna().sum() < 100 or f.nunique() < 2:
            continue
        try:
            rep = evaluate(f, y_t)
        except Exception as e:
            logger.debug("因子 %s 评估失败：%s", col, e)
            continue
        rows.append({"factor": col, "ic": rep["ic_mean"], "icir": rep["icir"],
                     "t_stat": rep["t_stat"], "win_rate": rep["ic_win_rate"],
                     "n_days": rep["n_days"]})
    if not rows:
        return pd.DataFrame(columns=["factor", "ic", "icir", "t_stat",
                                     "win_rate", "n_days"])
    df = pd.DataFrame(rows)
    return (df.reindex(df["ic"].abs().sort_values(ascending=False).index)
            .head(top_k).reset_index(drop=True))
