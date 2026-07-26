"""策略稳健性验证（v0.28 新增）。

**为什么必须有这个**

``eq backtest --sweep`` 是在**一只标的、一段区间**上跑出来的。这种结果
基本没有决策价值：

- 17 个策略在茅台 2024-07~2026-07 上多数为负，不代表策略不行，
  只代表这段行情不适合它们；
- 反过来，某个策略在这段跑出夏普 +1.14，也极可能只是**运气**——
  在 17 个策略里挑最好的那个，本身就是一次多重比较，
  纯噪声下也会有一个"看起来很好"。

ML 层在 v0.25 已经修过同一类问题（验证集既选模型又报成绩、没有 purge）。
策略层这里补齐对应的东西：

- :func:`multi_symbol`      同一策略在 N 只标的上的**分布**（中位数、盈利占比、最差）
- :func:`walk_forward`      滚动窗口样本外验证，段间 purge
- :func:`param_sweep`       参数网格扫描
- :func:`optimize`          **样本内选参 + 样本外验证**，并给出「参数高原」判断
- :func:`random_benchmark`  和随机信号比，回答「这策略比瞎买强吗」

判定原则和 ML 层一致：**选参数的集合不能同时用来报成绩**。
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from eq.backtest.metrics import compute_metrics
from eq.backtest.types import BacktestConfig
from eq.backtest.vectorized import VectorizedBacktester

logger = logging.getLogger(__name__)

SignalFunc = Callable[[pd.DataFrame], pd.Series]

# 判断策略好坏的默认指标。用夏普而非总收益：
# 总收益没有考虑波动，一个满仓死扛的策略在牛市里总收益永远最高。
DEFAULT_METRIC = "sharpe"


def run_once(df: pd.DataFrame, strategy: SignalFunc,
             cfg: BacktestConfig | None = None) -> dict[str, Any]:
    """跑一次回测，只要指标。失败返回全 0 而不是抛异常——
    批量评估里个别标的数据有问题不该中断整轮。"""
    try:
        res = VectorizedBacktester().run(df, strategy, cfg or BacktestConfig())
        return dict(res.metrics)
    except Exception as e:
        logger.debug("回测失败：%s", e)
        return compute_metrics(pd.Series(dtype=float), pd.DataFrame())


def _summarize(rows: list[dict[str, Any]], metric: str = DEFAULT_METRIC) -> dict[str, Any]:
    """把一组回测结果汇总成分布统计。

    **看中位数和盈利占比，不要只看均值**——均值会被一两个极端样本带偏，
    而策略稳不稳恰恰体现在"大多数情况下如何"。
    """
    vals = np.array([r.get(metric, 0.0) for r in rows], dtype="float64")
    vals = vals[np.isfinite(vals)]
    rets = np.array([r.get("total_return", 0.0) for r in rows], dtype="float64")
    rets = rets[np.isfinite(rets)]
    dds = np.array([r.get("max_drawdown", 0.0) for r in rows], dtype="float64")
    dds = dds[np.isfinite(dds)]
    if len(vals) == 0:
        return {"n": 0, "metric": metric}
    return {
        "n": int(len(vals)),
        "metric": metric,
        f"{metric}_median": float(np.median(vals)),
        f"{metric}_mean": float(vals.mean()),
        f"{metric}_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        f"{metric}_min": float(vals.min()),
        f"{metric}_max": float(vals.max()),
        "pct_positive": float((vals > 0).mean()),
        "return_median": float(np.median(rets)) if len(rets) else 0.0,
        "pct_profitable": float((rets > 0).mean()) if len(rets) else 0.0,
        "worst_drawdown": float(dds.min()) if len(dds) else 0.0,
        "total_trades": int(sum(r.get("num_trades", 0) for r in rows)),
    }


# ======================================================================
# 1) 多标的
# ======================================================================

def multi_symbol(
    strategy: SignalFunc,
    bars_by_symbol: dict[str, pd.DataFrame],
    cfg: BacktestConfig | None = None,
    metric: str = DEFAULT_METRIC,
) -> dict[str, Any]:
    """同一策略在多只标的上跑，看**分布**而不是单点。

    一个策略在 20 只票上有 14 只赚钱、夏普中位数 0.4，比在 1 只票上
    夏普 1.5 有说服力得多。
    """
    per: list[dict[str, Any]] = []
    for sym, df in bars_by_symbol.items():
        if df is None or len(df) < 30:
            continue
        m = run_once(df, strategy, cfg)
        m["symbol"] = sym
        m["bars"] = len(df)
        per.append(m)
    return {"per_symbol": per, "summary": _summarize(per, metric)}


# ======================================================================
# 2) Walk-Forward
# ======================================================================

@dataclass
class Window:
    train: slice
    test: slice
    train_range: tuple[Any, Any] = field(default=(None, None))
    test_range: tuple[Any, Any] = field(default=(None, None))


def walk_forward_windows(
    df: pd.DataFrame,
    n_splits: int = 5,
    train_bars: int = 250,
    test_bars: int = 60,
    embargo_bars: int = 5,
    expanding: bool = True,
) -> list[Window]:
    """切滚动窗口。段间留 ``embargo_bars`` 根空档。

    为什么要 embargo：策略用的指标（如 60 日均线）在训练段末尾和测试段
    开头是**同一批数据算出来的**，紧邻切分会让"样本外"沾到样本内的信息。
    """
    n = len(df)
    out: list[Window] = []
    for k in range(n_splits):
        test_end = n - k * test_bars
        test_start = test_end - test_bars
        train_end = test_start - embargo_bars
        if test_start <= 0 or train_end < train_bars:
            break
        train_start = 0 if expanding else max(0, train_end - train_bars)
        out.append(Window(
            train=slice(train_start, train_end),
            test=slice(test_start, test_end),
            train_range=(df.index[train_start], df.index[train_end - 1]),
            test_range=(df.index[test_start], df.index[test_end - 1]),
        ))
    out.reverse()
    return out


def walk_forward(
    df: pd.DataFrame,
    strategy: SignalFunc,
    n_splits: int = 5,
    train_bars: int = 250,
    test_bars: int = 60,
    embargo_bars: int = 5,
    cfg: BacktestConfig | None = None,
    metric: str = DEFAULT_METRIC,
) -> dict[str, Any]:
    """滚动前向验证：每个窗口只在**测试段**上评估。

    固定参数的策略在这里主要检验「跨时段稳定性」——
    某段特别好、其余全平，说明结果靠的是运气。
    """
    wins = walk_forward_windows(df, n_splits, train_bars, test_bars, embargo_bars)
    rows: list[dict[str, Any]] = []
    for i, w in enumerate(wins, 1):
        # 用「训练段 + 测试段」一起喂给策略，但只统计测试段的表现——
        # 技术指标需要前置数据才成形，只喂测试段的话前 N 根全是 NaN
        seg = df.iloc[w.train.start:w.test.stop]
        try:
            res = VectorizedBacktester().run(seg, strategy, cfg or BacktestConfig())
            eq = res.equity_curve.iloc[-(w.test.stop - w.test.start):]
            tr = res.trades
            if not tr.empty and "exit_date" in tr.columns:
                lo = df.index[w.test.start]
                tr = tr[pd.to_datetime(tr["exit_date"]) >= lo]
            m = compute_metrics(eq, tr)
        except Exception as e:
            logger.debug("walk-forward 窗口 %d 失败：%s", i, e)
            continue
        m.update(window=i,
                 test_start=str(pd.Timestamp(w.test_range[0]).date()),
                 test_end=str(pd.Timestamp(w.test_range[1]).date()))
        rows.append(m)
    return {"windows": rows, "summary": _summarize(rows, metric)}


# ======================================================================
# 3) 参数扫描 + 样本外优化
# ======================================================================

def expand_grid(grid: dict[str, Sequence]) -> list[dict[str, Any]]:
    """``{"fast":[5,10],"slow":[20,30]}`` → 4 个参数组合。"""
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, combo, strict=True))
            for combo in itertools.product(*(grid[k] for k in keys))]


def param_sweep(
    df: pd.DataFrame,
    factory: Callable[..., SignalFunc],
    grid: dict[str, Sequence],
    cfg: BacktestConfig | None = None,
    metric: str = DEFAULT_METRIC,
) -> pd.DataFrame:
    """参数网格扫描，返回每组参数的指标表（按 metric 降序）。"""
    combos = expand_grid(grid)
    if not combos:
        raise ValueError("参数网格为空")
    rows = []
    for i, params in enumerate(combos):
        try:
            m = run_once(df, factory(**params), cfg)
        except Exception as e:
            logger.debug("参数 %s 构造/回测失败：%s", params, e)
            continue
        rows.append({"_combo": i, **params,
                     **{k: v for k, v in m.items() if not isinstance(v, (list, dict))}})
    if not rows:
        raise ValueError("所有参数组合都失败了")
    return pd.DataFrame(rows).sort_values(metric, ascending=False).reset_index(drop=True)


def plateau_score(sweep: pd.DataFrame, param_names: Sequence[str],
                  metric: str = DEFAULT_METRIC, top_frac: float = 0.2) -> float:
    """「参数高原」得分 0~1：最优参数周围是不是也不错。

    这是**判断参数过拟合最实用的一招**。如果最优参数是孤立尖峰
    （旁边一格就掉下去），那多半是拟合了噪声，实盘一定失效；
    如果最优点周围是一片高地（高原），才说明这个参数区间是真有效的。

    做法：取表现前 ``top_frac`` 的参数组，看它们在参数空间里是否彼此靠近。
    返回 1 = 完美高原，0 = 孤立尖峰。
    """
    if sweep.empty or not param_names:
        return 0.0
    usable = [p for p in param_names if p in sweep.columns
              and pd.api.types.is_numeric_dtype(sweep[p])]
    if not usable:
        return 0.0
    k = max(2, int(len(sweep) * top_frac))
    top = sweep.nlargest(k, metric)
    # 各参数维度归一化后算 top 组的离散度，越集中越像高原
    spreads = []
    for p in usable:
        col = sweep[p].astype("float64")
        rng = col.max() - col.min()
        if rng <= 0:
            continue
        spreads.append(top[p].astype("float64").std(ddof=0) / rng)
    if not spreads:
        return 0.0
    # 随机分布时归一化标准差约 0.29（均匀分布），映射到 0
    return float(np.clip(1.0 - float(np.mean(spreads)) / 0.29, 0.0, 1.0))


def optimize(
    df: pd.DataFrame,
    factory: Callable[..., SignalFunc],
    grid: dict[str, Sequence],
    *,
    test_ratio: float = 0.3,
    embargo_bars: int = 5,
    cfg: BacktestConfig | None = None,
    metric: str = DEFAULT_METRIC,
) -> dict[str, Any]:
    """**样本内选参 + 样本外验证**的参数优化。

    直接在全样本上网格搜索再报最优值，是策略研究里最常见的自欺方式——
    等价于 ML 里"验证集既选模型又报成绩"。这里把数据切两段：
    前段选参数，后段（模型没见过）报成绩。

    Returns:
        ``{best_params, in_sample, out_of_sample, degradation, plateau,
        sweep, verdict}``。``degradation`` 是样本外相对样本内的衰减比例，
        衰减超过 50% 基本可以判定过拟合。
    """
    n = len(df)
    cut = int(n * (1 - test_ratio))
    if cut < 60 or n - cut < 30:
        raise ValueError(f"样本太短（{n} 根）无法切分参数优化的样本内/外")
    is_df = df.iloc[:cut]
    oos_df = df.iloc[max(0, cut - embargo_bars):]   # 留前置数据让指标成形

    combos = expand_grid(grid)
    sweep = param_sweep(is_df, factory, grid, cfg, metric)
    best = sweep.iloc[0]
    param_names = list(grid)
    # 从原始 combos 里按下标取回参数，保住原始类型——
    # 从 DataFrame 里读会把 int 变成 numpy.float64（5 → 5.0），
    # 再传给 range()/rolling() 这类只收整数的地方就会炸
    best_params = dict(combos[int(best["_combo"])])

    is_metric = float(best[metric])
    oos = run_once(oos_df, factory(**best_params), cfg)
    oos_metric = float(oos.get(metric, 0.0))

    if is_metric > 0:
        degradation = (is_metric - oos_metric) / abs(is_metric)
    else:
        degradation = 0.0 if oos_metric >= is_metric else 1.0
    plateau = plateau_score(sweep, param_names, metric)

    return {
        "best_params": best_params,
        "in_sample": {k: v for k, v in best.items() if k not in param_names},
        "out_of_sample": oos,
        "in_sample_metric": is_metric,
        "out_of_sample_metric": oos_metric,
        "degradation": float(degradation),
        "plateau": plateau,
        "n_combos": int(len(sweep)),
        "sweep": sweep,
        "verdict": optimize_verdict(is_metric, oos_metric, degradation, plateau),
    }


def optimize_verdict(is_metric: float, oos_metric: float,
                     degradation: float, plateau: float) -> str:
    """给一句「这组参数能不能用」的直白判断。"""
    if is_metric <= 0:
        return "样本内就没跑赢，参数再调也没用"
    if oos_metric <= 0:
        return f"样本外为负（{oos_metric:+.2f}），典型过拟合"
    if degradation > 0.5:
        return f"样本外衰减 {degradation:.0%}，过拟合嫌疑大"
    if plateau < 0.3:
        return f"最优参数是孤立尖峰（高原分 {plateau:.2f}），换个市场大概率失效"
    if degradation > 0.3:
        return f"样本外衰减 {degradation:.0%}，勉强可用但要盯着"
    return f"样本外保持住了（衰减 {degradation:.0%}，高原分 {plateau:.2f}）"


# ======================================================================
# 4) 随机基准
# ======================================================================

def random_benchmark(
    df: pd.DataFrame,
    strategy: SignalFunc,
    n_trials: int = 200,
    cfg: BacktestConfig | None = None,
    metric: str = DEFAULT_METRIC,
    seed: int = 42,
) -> dict[str, Any]:
    """和「同频率随机进出场」比，回答：这策略比瞎买强吗？

    做法：先跑真策略拿到交易次数和平均持仓时长，再生成 ``n_trials`` 组
    **交易频率匹配**的随机信号，看真策略的指标排在随机分布的什么位置。

    频率必须匹配——不然拿一个高频策略去比低频随机基准，差异全来自
    交易成本而不是选时能力。

    Returns:
        ``{actual, random_mean, random_std, percentile, p_value, n_trials}``。
        ``p_value`` 是随机策略跑赢真策略的比例，< 0.05 才算显著。
    """
    from eq.strategy import BUY, HOLD, SELL

    cfg = cfg or BacktestConfig()
    actual = run_once(df, strategy, cfg)
    actual_metric = float(actual.get(metric, 0.0))
    n_trades = max(int(actual.get("num_trades", 0)), 1)

    n = len(df)
    rng = np.random.default_rng(seed)
    rand_metrics: list[float] = []
    for _ in range(n_trials):
        sig = pd.Series(HOLD, index=df.index)
        # 随机挑 n_trades 对进出场点，保证进场在出场之前
        picks = np.sort(rng.choice(n, size=min(2 * n_trades, n), replace=False))
        for j in range(0, len(picks) - 1, 2):
            sig.iloc[picks[j]] = BUY
            sig.iloc[picks[j + 1]] = SELL
        m = run_once(df, lambda d, _s=sig: _s.reindex(d.index).fillna(HOLD), cfg)
        v = float(m.get(metric, 0.0))
        if np.isfinite(v):
            rand_metrics.append(v)

    arr = np.array(rand_metrics) if rand_metrics else np.array([0.0])
    beat = float((arr >= actual_metric).mean())
    return {
        "metric": metric,
        "actual": actual_metric,
        "random_mean": float(arr.mean()),
        "random_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "percentile": float((arr < actual_metric).mean() * 100),
        "p_value": beat,
        "n_trials": len(arr),
        "n_trades": n_trades,
        "verdict": ("显著优于随机" if beat < 0.05 else
                    "略优于随机但不显著" if beat < 0.3 else
                    "与随机无异——这个结果不值得相信"),
    }


# ======================================================================
# 格式化
# ======================================================================

def format_multi_symbol(report: dict[str, Any], top: int = 10) -> str:
    s = report["summary"]
    if not s.get("n"):
        return "\n多标的评估：无有效样本\n"
    m = s["metric"]
    lines = [f"\n多标的稳健性（{s['n']} 只标的）"]
    lines.append(
        f"  {m} 中位数 {s[f'{m}_median']:+.2f}   均值 {s[f'{m}_mean']:+.2f}"
        f"   区间 [{s[f'{m}_min']:+.2f}, {s[f'{m}_max']:+.2f}]"
    )
    lines.append(
        f"  盈利标的占比 {s['pct_profitable']:.0%}   收益中位数 {s['return_median']:+.2%}"
        f"   最差回撤 {s['worst_drawdown']:+.2%}   总交易 {s['total_trades']} 笔"
    )
    per = sorted(report["per_symbol"], key=lambda r: r.get(m, 0), reverse=True)
    lines.append(f"\n  {'标的':<14}{'总收益':>10}{'夏普':>8}{'回撤':>9}{'交易':>6}")
    lines.append("  " + "-" * 48)
    for r in per[:top]:
        lines.append(f"  {r['symbol']:<14}{r.get('total_return', 0):>+9.2%}"
                     f"{r.get('sharpe', 0):>+8.2f}{r.get('max_drawdown', 0):>+9.2%}"
                     f"{r.get('num_trades', 0):>6}")
    if len(per) > top:
        lines.append(f"  …（共 {len(per)} 只）")
    return "\n".join(lines) + "\n"


def format_walk_forward(report: dict[str, Any]) -> str:
    rows = report["windows"]
    s = report["summary"]
    if not rows:
        return "\nWalk-Forward：数据不足以切出窗口\n"
    m = s["metric"]
    lines = [f"\nWalk-Forward 样本外（{len(rows)} 个窗口）"]
    lines.append(f"  {'窗口':<6}{'测试区间':<26}{'总收益':>10}{'夏普':>8}{'回撤':>9}{'交易':>6}")
    lines.append("  " + "-" * 66)
    for r in rows:
        lines.append(f"  {r['window']:<6}{r['test_start']}~{r['test_end']:<12}"
                     f"{r.get('total_return', 0):>+9.2%}{r.get('sharpe', 0):>+8.2f}"
                     f"{r.get('max_drawdown', 0):>+9.2%}{r.get('num_trades', 0):>6}")
    lines.append("  " + "-" * 66)
    lines.append(
        f"  {m} 中位数 {s[f'{m}_median']:+.2f}   为正的窗口占比 {s['pct_positive']:.0%}"
        f"   窗口间标准差 {s[f'{m}_std']:.2f}"
    )
    lines.append(f"  判定：{walk_forward_verdict(s)}")
    return "\n".join(lines) + "\n"


def walk_forward_verdict(summary: dict[str, Any]) -> str:
    n = summary.get("n", 0)
    if n < 3:
        return f"窗口太少（{n} 个），结论不可信"
    m = summary["metric"]
    med = summary[f"{m}_median"]
    pos = summary["pct_positive"]
    std = summary[f"{m}_std"]
    if med <= 0:
        return f"多数窗口不赚钱（中位数 {med:+.2f}），策略在这个标的上不成立"
    if pos < 0.5:
        return f"只有 {pos:.0%} 的窗口为正，靠个别窗口撑起来的"
    if std > abs(med) * 2:
        return f"窗口间波动过大（std {std:.2f} vs 中位数 {med:.2f}），不稳定"
    return f"跨窗口稳定（中位数 {med:+.2f}，{pos:.0%} 窗口为正）"
