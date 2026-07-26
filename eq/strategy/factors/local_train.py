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
           "load_local_model", "predict_local", "factor_scan",
           "baseline_composite", "score_matrix_local", "backtest_local"]


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
        test_report = evaluate(pred, y_test, horizon=horizon)
    ic = test_report["ic_mean"] if test_report else valid_ic

    import pickle as _pkl

    from eq.strategy.factors.ml_workflow import _ensure_dir

    suffix = f"_x{n_seeds}" if n_seeds > 1 else ""
    model_path = _ensure_dir() / f"local_{algo}_{horizon}d{suffix}.pkl"
    with open(model_path, "wb") as f:
        # split_bounds 必须一起存：回测时要知道测试段从哪天开始，
        # 否则会滑进训练段——样本内的权益曲线又漂亮又没有意义
        _pkl.dump({"model": model, "pipeline": pipe,
                   "features": list(x.columns), "horizon": horizon,
                   "split_bounds": split.bounds, "label_norm": label_norm}, f)

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
        t_nw / win_rate / n_days``。``t_nw`` 是 Newey-West 修正后的 t 值，
        扣掉了重叠标签造成的自相关——**该看的是它，不是 t_stat**。
        IC 为负的因子反向用同样有效，所以排序看绝对值，``ic`` 列保留原始符号。
    """

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

    df = daily_ic_matrix(x_t, y_t)
    if df.empty:
        return pd.DataFrame(columns=["factor", "ic", "icir", "t_stat", "t_nw",
                                     "win_rate", "n_days"])
    stats = summarize_ic_matrix(df, horizon=horizon)
    return (stats.reindex(stats["ic"].abs().sort_values(ascending=False).index)
            .head(top_k).reset_index(drop=True))


def daily_ic_matrix(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """一次算出**所有因子**的每日横截面 Rank IC，返回 ``日期 × 因子`` 矩阵。

    改之前是 158 个因子各调一次 :func:`evaluation.evaluate`，而 ``evaluate``
    每次要跑三遍数据（Rank IC + 分层收益 + Pearson IC），每遍都按日 groupby。
    300 只票的实测规模上要跑十几分钟，用户等到以为卡死了。

    这里只算需要的那一样，并且把 158 列一起做：

    - ``groupby(日期).rank()`` 一次把整张表按日排名（158 列同时）
    - 逐日去均值后，Rank IC = 中心化秩的相关，可以写成几个 groupby 求和

    数值上和逐列跑 ``spearman`` 一致，但快两个数量级。

    **缺失值处理**：只保留特征全非空的行（Alpha158 前 60 根是暖机期，
    必然有 NaN）。保留比例会打日志——比例过低说明历史长度不够。
    """
    if len(x) == 0 or len(y) == 0 or x.shape[1] == 0:
        return pd.DataFrame()
    if "datetime" not in (x.index.names or []):
        # 没有日期层就算不了「横截面」IC，这是调用方丢了索引，直接说清楚
        raise ValueError("特征面板缺少 datetime 索引层，无法计算横截面 IC")
    dates = x.index.get_level_values("datetime")
    keep = x.notna().all(axis=1) & y.notna()
    if keep.sum() == 0:
        return pd.DataFrame()
    if keep.mean() < 0.99:
        logger.info("单因子扫描保留 %.1f%% 的行（其余含 NaN，多为暖机期）",
                    keep.mean() * 100)
    xs, ys, ds = x[keep], y[keep], dates[keep]

    # 每天至少要有这么多只票才算得出有意义的横截面相关
    from eq.strategy.factors.evaluation import MIN_STOCKS_PER_DAY

    counts = ys.groupby(ds).transform("size")
    ok = counts >= MIN_STOCKS_PER_DAY
    xs, ys, ds = xs[ok.to_numpy()], ys[ok.to_numpy()], ds[ok.to_numpy()]
    if len(ys) == 0:
        return pd.DataFrame()

    g = ds.to_numpy()
    rx = xs.groupby(g).rank()
    ry = ys.groupby(g).rank()
    cx = rx - rx.groupby(g).transform("mean")
    cy = ry - ry.groupby(g).transform("mean")

    num = cx.mul(cy, axis=0).groupby(g).sum()
    sxx = (cx * cx).groupby(g).sum()
    syy = (cy * cy).groupby(g).sum()
    den = np.sqrt(sxx.mul(syy, axis=0))
    ic = num / den.replace(0, np.nan)
    # 当日预测或标签无区分度（分母 0）时记 0（中性），和 daily_ic 的口径一致
    return ic.fillna(0.0)


def summarize_ic_matrix(ic: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """把 ``日期 × 因子`` 的 IC 矩阵汇总成每因子一行的指标表。

    ``t_nw`` 是 Newey-West 修正后的 t 值（滞后阶 ``horizon-1``），
    扣掉重叠标签造成的自相关——**该看的是它，不是 t_stat**。
    """
    from eq.strategy.factors.evaluation import _newey_west_t

    n = len(ic)
    mean = ic.mean()
    std = ic.std(ddof=1) if n > 1 else pd.Series(0.0, index=ic.columns)
    std = std.where(std >= 1e-12, 0.0)
    icir = (mean / std).where(std > 0, 0.0)
    return pd.DataFrame({
        "factor": ic.columns,
        "ic": mean.to_numpy(),
        "icir": icir.to_numpy(),
        "t_stat": (icir * np.sqrt(n)).where(std > 0, 0.0).to_numpy(),
        "t_nw": [_newey_west_t(ic[c].dropna(), horizon) for c in ic.columns],
        "win_rate": (ic > 0).mean().to_numpy(),
        "n_days": n,
    })


def baseline_composite(
    symbols: list[str],
    *,
    horizon: int = 5,
    days: int = 1200,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    embargo_days: int | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """无参数基准：在**验证段**挑因子，在**测试段**评估，和模型成绩直接可比。

    :func:`factor_scan` 是在测试段上挑最大值，带选择偏差，只能当粗略参照。
    这个函数把选择和评估分开——用验证段（模型早停也用它）排出 |IC| 前 k 个因子，
    按各自 IC 的符号取向后**等权合成**，然后在测试段量一次。全程没有拟合参数。

    这是判断「模型到底有没有用」最干净的对照：

    - 模型 test IC 明显高于合成基准 → 训练确实创造了价值
    - 打平或更低 → 158 维模型没跑赢一个等权公式。可能是过拟合到训练段的旧行情，
      也可能是这批特征本来就只有那么点信息——两者都说明不该用这个模型交易。

    Returns:
        ``{"selected": [...], "composite": {...}, "singles": DataFrame,
        "n_test_days": int}``；``composite`` 是 :func:`evaluation.evaluate` 的报告。
    """
    from eq.strategy.factors.evaluation import evaluate

    embargo = horizon if embargo_days is None else int(embargo_days)
    bars = load_bars(symbols, days=days)
    if not bars:
        raise ValueError("一只标的的行情都没拉到")
    x, y = build_dataset(bars, horizon=horizon)
    split = purged_split(x.index, valid_ratio=valid_ratio, test_ratio=test_ratio,
                         embargo_days=embargo, with_test=test_ratio > 0)
    if split.test is None or split.test.sum() == 0:
        raise ValueError("没有独立测试段，无法做无偏对照（test_ratio 不能为 0）")

    x_v, y_v = x[split.valid], y[split.valid]
    x_t, y_t = x[split.test], y[split.test]

    # 1) 在验证段给每个因子打分（选择只用验证段，测试段完全不参与）。
    #    走 daily_ic_matrix 一次算完 158 列——逐列调 evaluate 在 300 只票的
    #    规模上要几分钟，而这里只需要 IC 均值。
    ic_v = daily_ic_matrix(x_v, y_v)
    if ic_v.empty:
        raise ValueError("验证段上没有可用因子")
    means = ic_v.mean().dropna()
    if means.empty:
        raise ValueError("验证段上没有可用因子")
    picked = [(c, float(means[c]))
              for c in means.abs().sort_values(ascending=False).index[:top_k]]

    # 2) 按验证段 IC 的符号取向，逐因子做截面 rank 归一化后等权相加。
    #    先归一化再相加是必须的：158 个因子的量纲天差地别，
    #    直接加等于让方差最大的那个说了算。
    parts = []
    for col, ic in picked:
        z = pp.cs_rank_norm(x_t[col])
        parts.append(z * (1.0 if ic >= 0 else -1.0))
    composite = sum(parts) / len(parts)

    # 3) 在测试段评估：合成基准 + 各成分单独表现（都用同一把尺）
    singles = []
    for col, vic in picked:
        rep = evaluate(x_t[col], y_t, horizon=horizon)
        singles.append({"factor": col, "valid_ic": vic, "test_ic": rep["ic_mean"],
                        "test_icir": rep["icir"], "test_t": rep["t_stat"]})
    return {
        "selected": [c for c, _ in picked],
        "composite": evaluate(composite, y_t, horizon=horizon),
        "singles": pd.DataFrame(singles),
        "n_test_days": int(len(set(x_t.index.get_level_values("datetime")))),
    }


# ======================================================================
# 组合回测：把 IC 换算成钱
# ======================================================================

def _test_start(blob: dict[str, Any], model_id: str) -> pd.Timestamp:
    """确定回测起点＝测试段第一天。定不下来就报错，**绝不猜**。

    在训练段上回测会给出一条又漂亮又毫无意义的权益曲线——模型见过那段数据。
    这种错误不会报警、不会崩，只会让人高兴，所以宁可拒绝执行。
    """
    b = blob.get("split_bounds") or {}
    if b.get("test"):
        return pd.Timestamp(b["test"][0])
    # 老模型没存 split_bounds，退回注册表里的 valid 区间末尾 + embargo
    from eq.db import execute

    rows = execute("SELECT valid_period, metrics FROM ml_models WHERE id = ?", (model_id,))
    if rows and rows[0]["valid_period"] and "~" in rows[0]["valid_period"]:
        import json

        end = rows[0]["valid_period"].split("~")[1].strip()
        emb = int(json.loads(rows[0]["metrics"] or "{}").get("embargo_days", 5))
        if end:
            return pd.Timestamp(end) + pd.Timedelta(days=emb)
    raise ValueError(
        f"定不下 {model_id} 的测试段起点（模型太老、没存 split_bounds）。"
        "重训一次即可；或者显式传 start= 指定从哪天开始回测。"
        "注意起点必须晚于训练/验证段，否则是样本内回测。")


def score_matrix_local(
    model_id: str,
    symbols: list[str],
    *,
    days: int = 1200,
    start: str | None = None,
    top_n: int = 10,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """用本地模型给**每一个交易日**打分，返回 ``(日期 × 标的 分数矩阵, 行情)``。

    只输出测试段之后的日期——模型见过训练段和验证段，在那上面回测没有意义。

    分数矩阵的取值：当日排名前 ``top_n`` 的标的取 ``1.0 → 0.5`` 的线性递减权重，
    其余为 0。组合引擎按 ``>0`` 判断是否想持有、按数值大小分配权重。
    """
    blob = load_local_model(model_id)
    model, pipe, feats = blob["model"], blob["pipeline"], blob["features"]

    bars = load_bars(symbols, days=days)
    if not bars:
        raise ValueError("一只标的的行情都没拉到")
    x = alpha_mod.alpha158(bars)
    if x.empty:
        raise ValueError("特征为空——行情根数不够（每只至少 70 根）")

    begin = pd.Timestamp(start) if start else _test_start(blob, model_id)
    dates = x.index.get_level_values("datetime")
    x = x[dates >= begin]
    if x.empty:
        raise ValueError(f"{begin.date()} 之后没有数据可回测")
    missing = [c for c in feats if c not in x.columns]
    if missing:
        raise ValueError(f"特征对不上，缺 {missing[:5]}——模型需要重训")

    scores = pd.Series(np.asarray(model.predict(pipe.transform(x[feats]))).reshape(-1),
                       index=x.index)
    wide = scores.unstack("instrument")

    # 逐日取前 top_n，权重从 1.0 线性降到 0.5——让引擎知道谁更被看好，
    # 又不至于让第一名一家独大（真正的仓位上限由 PortfolioConfig 管）
    ranks = wide.rank(axis=1, ascending=False, method="first")
    sel = ranks <= top_n
    weight = (1.0 - 0.5 * (ranks - 1) / max(1, top_n - 1)).where(sel, 0.0)
    return weight.fillna(0.0), bars


def backtest_local(
    model_id: str,
    symbols: list[str],
    *,
    days: int = 1200,
    start: str | None = None,
    top_n: int = 10,
    cfg=None,
) -> dict[str, Any]:
    """把本地模型**真的跑一遍组合回测**——含 A 股真实成本。

    这是回答「IC 0.011 到底是赚是亏」的唯一办法。IC 不含手续费、不含印花税、
    不含最低佣金、不含换手，也不含「只能买整手」这种约束。
    日频 IC 0.01 这个量级，成本大概率把它整个吃掉——但吃掉多少要跑出来才知道。

    Returns:
        ``{"result", "gross", "cost_drag", "start", "n_symbols", "n_days"}``。
        ``gross`` 是同信号同约束、但**零成本**的对照，``cost_drag`` 是两者的
        总收益之差——即交易成本实际吃掉了多少。
    """
    from eq.backtest.portfolio import PortfolioConfig, run_portfolio

    weight, bars = score_matrix_local(model_id, symbols, days=days,
                                      start=start, top_n=top_n)
    cfg = cfg or PortfolioConfig(max_positions=top_n, rebalance="weekly",
                                 allocation="score", cost_model="a_share")
    # 只保留有分数的那段行情，否则组合引擎会从更早的日期开始空转
    begin = weight.index.min()
    sub = {s: d[d.index >= begin] for s, d in bars.items() if s in weight.columns}
    sub = {s: d for s, d in sub.items() if len(d) >= 30}
    if not sub:
        raise ValueError("测试段太短，组合回测至少要 30 根 bar")
    w = weight.reindex(columns=list(sub)).fillna(0.0)
    res = run_portfolio(sub, w, cfg)

    # 零成本对照：同一套信号、同一套约束，只把交易成本抹掉。
    # 差值就是成本真实吃掉了多少——日频选股的换手动辄几十倍，
    # 印花税 0.1% + 佣金在这个量级上是决定性的，不量出来只能靠猜。
    import dataclasses

    free_cfg = dataclasses.replace(cfg, cost_model=None,
                                   commission_bps=0.0, slippage_bps=0.0)
    gross = run_portfolio(sub, w, free_cfg)
    net_ret = res.metrics.get("total_return", 0.0)
    gross_ret = gross.metrics.get("total_return", 0.0)
    return {"result": res, "gross": gross, "start": str(begin.date()),
            "n_symbols": len(sub), "n_days": len(weight),
            "cost_drag": gross_ret - net_ret}
