"""组合级回测（v0.29 新增）—— 从「单标的」走到「一个账户管一篮子」。

**为什么必须有**

之前所有回测都是单标的：一只票、满仓/空仓。这和散户实际的用法差得很远：

- 真实账户是**一笔钱同时管十几只票**，资金怎么分配是核心问题
  （等权？按波动率反比？按信号强弱？）
- 单票回测算不出**分散化收益**——组合的波动远小于成分股波动的平均
- 单票回测也看不到**资金约束**：信号同时看多 20 只但只有 10 万块，买哪些？
- 换仓成本在组合层面完全不同：调仓是一堆小额买卖，正好撞上
  :mod:`eq.backtest.cost` 里说的最低佣金问题

本模块提供：

- 三种资金分配：等权 / 波动率反比（风险平价的简化版）/ 信号强弱加权
- 约束：最大持仓数、单票权重上限、现金缓冲
- 调仓节奏：信号驱动 / 周度 / 月度
- 真实成本：复用 :class:`~eq.backtest.cost.CostModel`（含最低佣金、印花税）
- 输出：组合权益曲线 + 逐标的贡献 + 换手率 + 完整绩效指标

**关于前视偏差**：本引擎对每只标的**一次性**算出整条信号序列，而不是逐日
重算。这要求信号函数本身是因果的（只用当期及历史数据）——本项目因子库里
所有 rolling/ewm/shift 都满足，自己写策略时也必须保证这点。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from eq.backtest.cost import CostModel, get_cost_model
from eq.backtest.metrics import compute_metrics
from eq.strategy import BUY, HOLD, SELL

logger = logging.getLogger(__name__)

SignalFunc = Callable[[pd.DataFrame], pd.Series]
Allocation = Literal["equal", "inverse_vol", "score"]
Rebalance = Literal["signal", "daily", "weekly", "monthly"]


@dataclass
class PortfolioConfig:
    """组合回测配置。"""

    initial_cash: float = 1_000_000.0
    max_positions: int = 10          # 最多同时持有几只
    max_weight: float = 0.25         # 单票权重上限——防止组合被一只票绑架
    min_weight: float = 0.02         # 低于此权重不值得买（会被最低佣金吃掉）
    cash_buffer: float = 0.02        # 留出的现金比例，应对滑点和整手取整
    allocation: Allocation = "equal"
    rebalance: Rebalance = "signal"
    vol_period: int = 20             # inverse_vol 用的波动率窗口
    cost_model: str | CostModel | None = "a_share"
    commission_bps: float = 2.5      # cost_model=None 时的回退
    slippage_bps: float = 5.0
    # 执行延迟（bar）：0 = 收盘看到信号、同一根收盘成交（散户做不到）；
    # 1 = 次日成交。组合层面尤其重要——调仓要动一篮子票，更不可能在收盘瞬间完成。
    execution_delay: int = 1

    def resolve_costs(self) -> CostModel:
        from eq.backtest.cost import from_bps

        m = get_cost_model(self.cost_model)
        return m if m is not None else from_bps(self.commission_bps, self.slippage_bps)


@dataclass
class PortfolioResult:
    equity_curve: pd.Series
    weights: pd.DataFrame            # 逐日各标的权重
    trades: pd.DataFrame
    metrics: dict[str, Any] = field(default_factory=dict)
    contribution: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        m = self.metrics
        return (f"总收益 {m.get('total_return', 0):+.2%}  年化 {m.get('annual_return', 0):+.2%}  "
                f"夏普 {m.get('sharpe', 0):+.2f}  最大回撤 {m.get('max_drawdown', 0):+.2%}  "
                f"换手 {m.get('annual_turnover', 0):.1f}x/年  "
                f"平均持仓 {m.get('avg_positions', 0):.1f} 只  交易 {m.get('num_trades', 0)} 笔")


# ======================================================================
# 内部工具
# ======================================================================

def _signal_matrix(bars: dict[str, pd.DataFrame], strategy, 
                   index: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """一次性算出所有标的的信号，对齐到统一日期轴。

    ``strategy`` 支持三种形态：

    - 单个 ``SignalFunc``：对每只标的分别调用（技术策略的常规用法）
    - ``{symbol: SignalFunc}``：每只标的用各自的策略
    - ``DataFrame``（日期 × 标的的分数矩阵）：**横截面**策略直接给分数，
      ML 选股就属于这类——「买分数最高的 N 只」本质上不是逐标的的信号，
      而是一个截面排序问题

    Returns:
        ``(desire, score)``——``desire`` 是 0/1 的「是否想持有」，
        ``score`` 是 0~1 的信号强度（三态信号强度恒为 1）。
    """
    # 横截面分数矩阵：直接用
    if isinstance(strategy, pd.DataFrame):
        sc = strategy.reindex(index).ffill()
        sc = sc.reindex(columns=[c for c in sc.columns if c in bars])
        if sc.empty or not len(sc.columns):
            raise ValueError("分数矩阵与行情的标的没有交集")
        return (sc > 0).astype("float64").fillna(0.0), sc.clip(0.0, 1.0).fillna(0.0)

    per_symbol = isinstance(strategy, dict)
    desire, score = {}, {}
    for sym, df in bars.items():
        fn = strategy.get(sym) if per_symbol else strategy
        if fn is None:
            continue
        try:
            raw = pd.Series(fn(df)).reindex(df.index)
        except Exception as e:
            logger.debug("标的 %s 信号计算失败：%s", sym, e)
            continue
        if raw.dtype == object or raw.isin([BUY, SELL, HOLD]).any():
            state = pd.Series(np.nan, index=df.index, dtype="float64")
            state[raw == BUY] = 1.0
            state[raw == SELL] = 0.0
            want = state.ffill().fillna(0.0)
            stren = want
        else:
            num = pd.to_numeric(raw, errors="coerce").fillna(0.0)
            want = (num > 0).astype("float64")
            stren = num.clip(0.0, 1.0)
        desire[sym] = want.reindex(index).ffill().fillna(0.0)
        score[sym] = stren.reindex(index).ffill().fillna(0.0)
    if not desire:
        raise ValueError("所有标的的信号都算不出来")
    return pd.DataFrame(desire).fillna(0.0), pd.DataFrame(score).fillna(0.0)


def _inverse_vol_weights(bars: dict[str, pd.DataFrame], index: pd.DatetimeIndex,
                         period: int) -> pd.DataFrame:
    """波动率反比权重原料：1/波动。波动大的少配，是风险平价的简化版。"""
    out = {}
    for sym, df in bars.items():
        ret = df["close"].pct_change()
        vol = ret.rolling(period).std()
        inv = 1.0 / vol.replace(0, np.nan)
        out[sym] = inv.reindex(index).ffill()
    w = pd.DataFrame(out)
    # 全 NaN 的早期用等权兜底
    return w.fillna(w.mean(axis=1).to_frame().to_numpy().repeat(w.shape[1], axis=1).mean()).fillna(1.0)


def _rebalance_mask(index: pd.DatetimeIndex, mode: Rebalance,
                    desire: pd.DataFrame) -> np.ndarray:
    """哪些日子要调仓。"""
    if mode == "daily":
        return np.ones(len(index), dtype=bool)
    if mode in ("weekly", "monthly"):
        s = pd.Series(index, index=index)
        key = s.dt.isocalendar().week if mode == "weekly" else s.dt.month
        year = s.dt.year
        tag = year.astype(str) + "-" + key.astype(str)
        return (tag != tag.shift(1)).to_numpy()
    # signal：持仓集合发生变化时才调
    changed = (desire != desire.shift(1)).any(axis=1).to_numpy()
    changed[0] = True
    return changed


def _target_weights(row_desire: np.ndarray, row_score: np.ndarray,
                    row_invvol: np.ndarray, cfg: PortfolioConfig) -> np.ndarray:
    """算一天的目标权重（已应用持仓数/权重上限/现金缓冲）。"""
    n = len(row_desire)
    w = np.zeros(n)
    idx = np.flatnonzero(row_desire > 0)
    if len(idx) == 0:
        return w

    # 超过最大持仓数时按信号强度取前 N
    if len(idx) > cfg.max_positions:
        idx = idx[np.argsort(-row_score[idx], kind="stable")[:cfg.max_positions]]

    if cfg.allocation == "equal":
        raw = np.ones(len(idx))
    elif cfg.allocation == "inverse_vol":
        raw = np.nan_to_num(row_invvol[idx], nan=0.0, posinf=0.0)
        if raw.sum() <= 0:
            raw = np.ones(len(idx))
    elif cfg.allocation == "score":
        raw = np.clip(row_score[idx], 0.0, None)
        if raw.sum() <= 0:
            raw = np.ones(len(idx))
    else:
        raise ValueError(f"未知分配方式 {cfg.allocation}")

    investable = max(0.0, 1.0 - cfg.cash_buffer)
    w[idx] = raw / raw.sum() * investable
    # 单票上限：截断后把溢出的部分按比例分给未触顶的
    for _ in range(5):
        over = w > cfg.max_weight
        if not over.any():
            break
        excess = (w[over] - cfg.max_weight).sum()
        w[over] = cfg.max_weight
        room = (w > 0) & ~over
        if not room.any():
            break
        w[room] += excess * (w[room] / w[room].sum())
    # 太小的仓位干脆不要——会被最低佣金吃掉
    w[(w > 0) & (w < cfg.min_weight)] = 0.0
    total = w.sum()
    if total > investable:
        w *= investable / total
    return w


# ======================================================================
# 主引擎
# ======================================================================

def run_portfolio(
    bars: dict[str, pd.DataFrame],
    strategy,
    cfg: PortfolioConfig | None = None,
) -> PortfolioResult:
    """跑一次组合回测。

    Args:
        bars: ``{symbol: OHLCV DataFrame}``，各标的日期不必完全一致
        strategy: 信号函数 / ``{symbol: 信号函数}`` / 日期×标的的分数 DataFrame。
            函数形态要求**因果**（只用当期及历史数据），本项目因子库都满足。
    """
    cfg = cfg or PortfolioConfig()
    bars = {s: d for s, d in bars.items() if d is not None and len(d) >= 30}
    if not bars:
        raise ValueError("没有足够长的标的数据（每只至少 30 根 bar）")

    index = pd.DatetimeIndex(sorted(set().union(*(d.index for d in bars.values()))))
    costs = cfg.resolve_costs()

    desire, score = _signal_matrix(bars, strategy, index)
    delay = max(0, int(cfg.execution_delay))
    if delay:
        desire = desire.shift(delay).fillna(0.0)
        score = score.shift(delay).fillna(0.0)
    syms = list(desire.columns)
    prices = pd.DataFrame(
        {s: bars[s]["close"].reindex(index).ffill() for s in syms}
    )
    # 停牌/未上市（价格还是 NaN）的不能交易
    tradable = pd.DataFrame(
        {s: bars[s]["close"].reindex(index).notna() for s in syms}
    ).ffill().fillna(False)
    invvol = _inverse_vol_weights({s: bars[s] for s in syms}, index, cfg.vol_period)[syms]
    rebal = _rebalance_mask(index, cfg.rebalance, desire)

    d_arr = desire.to_numpy()
    s_arr = score.to_numpy()
    v_arr = invvol.to_numpy()
    p_arr = prices.to_numpy()
    t_arr = tradable.to_numpy()

    n_days, n_sym = p_arr.shape
    shares = np.zeros(n_sym)
    cash = cfg.initial_cash
    entry_price = np.zeros(n_sym)
    entry_date: list[Any] = [None] * n_sym
    equity_hist = np.zeros(n_days)
    weight_hist = np.zeros((n_days, n_sym))
    trades: list[dict[str, Any]] = []
    turnover_hist = np.zeros(n_days)

    for i in range(n_days):
        px = p_arr[i]
        valid = np.isfinite(px) & (px > 0)
        mtm = float(np.nansum(np.where(valid, shares * px, 0.0)))
        equity = cash + mtm

        if rebal[i] and equity > 0:
            want = d_arr[i] * valid * t_arr[i]
            tw = _target_weights(want, s_arr[i], v_arr[i], cfg)
            target_shares = np.zeros(n_sym)
            for j in range(n_sym):
                if tw[j] > 0 and valid[j]:
                    target_shares[j] = costs.round_lots(equity * tw[j] / px[j])

            # 先卖后买：腾出现金再买入，否则会因现金不足少买
            for j in range(n_sym):
                if not valid[j] or not t_arr[i][j]:
                    continue
                delta = target_shares[j] - shares[j]
                if delta >= 0:
                    continue
                qty = min(-delta, shares[j])
                sell_px = px[j] * (1 - costs.slippage_rate)
                gross = qty * sell_px
                proceeds = gross - costs.trade_cost(gross, "sell")
                cash += proceeds
                shares[j] -= qty
                turnover_hist[i] += gross
                if shares[j] <= 0 and entry_price[j] > 0:
                    basis = qty * entry_price[j]
                    trades.append({
                        "symbol": syms[j], "entry_date": entry_date[j],
                        "exit_date": index[i], "entry_price": entry_price[j],
                        "exit_price": sell_px, "shares": qty,
                        "pnl": proceeds - (basis + costs.trade_cost(basis, "buy")),
                        "pnl_pct": (proceeds - basis - costs.trade_cost(basis, "buy")) / basis,
                    })
                    entry_price[j] = 0.0
                    entry_date[j] = None

            for j in range(n_sym):
                if not valid[j] or not t_arr[i][j]:
                    continue
                delta = target_shares[j] - shares[j]
                if delta <= 0:
                    continue
                buy_px = px[j] * (1 + costs.slippage_rate)
                qty = delta
                value = qty * buy_px
                total = value + costs.trade_cost(value, "buy")
                # 现金不够就按整手往下缩
                while qty > 0 and total > cash:
                    qty -= max(costs.lot_size, 1)
                    value = qty * buy_px
                    total = value + costs.trade_cost(value, "buy")
                if qty <= 0:
                    continue
                if shares[j] <= 0:
                    entry_price[j] = buy_px
                    entry_date[j] = index[i]
                else:
                    entry_price[j] = ((entry_price[j] * shares[j] + buy_px * qty)
                                      / (shares[j] + qty))
                cash -= total
                shares[j] += qty
                turnover_hist[i] += value

        mtm = float(np.nansum(np.where(valid, shares * px, 0.0)))
        equity = cash + mtm
        equity_hist[i] = equity
        if equity > 0:
            weight_hist[i] = np.where(valid, shares * px, 0.0) / equity

    eq = pd.Series(equity_hist, index=index, name="equity")
    weights = pd.DataFrame(weight_hist, index=index, columns=syms)
    trades_df = pd.DataFrame(trades)
    metrics = compute_metrics(eq, trades_df)

    # 组合特有指标
    years = max(len(index) / 252, 1e-9)
    avg_equity = float(eq[eq > 0].mean()) if (eq > 0).any() else cfg.initial_cash
    metrics["annual_turnover"] = float(turnover_hist.sum() / max(avg_equity, 1e-9) / years)
    metrics["avg_positions"] = float((weights > 1e-6).sum(axis=1).mean())
    metrics["max_positions_held"] = int((weights > 1e-6).sum(axis=1).max())
    metrics["n_symbols"] = len(syms)
    metrics["cash_pct_mean"] = float((1 - weights.sum(axis=1)).clip(0, 1).mean())
    # 已实现 / 未实现拆分：contribution 表只统计**平掉的**交易，
    # 期末还持着的仓位不在里面。不拆开的话会出现「逐标的贡献全是负的、
    # 总收益却是正的」这种看不懂的情况。
    realized = float(trades_df["pnl"].sum()) if not trades_df.empty else 0.0
    metrics["realized_pnl"] = realized
    metrics["unrealized_pnl"] = float(eq.iloc[-1] - cfg.initial_cash - realized) if len(eq) else 0.0
    metrics["open_positions_end"] = int((weight_hist[-1] > 1e-6).sum()) if n_days else 0

    # 逐标的贡献
    contrib = pd.DataFrame()
    if not trades_df.empty:
        g = trades_df.groupby("symbol")
        contrib = pd.DataFrame({
            "trades": g.size(),
            "total_pnl": g["pnl"].sum(),
            "win_rate": g["pnl_pct"].apply(lambda x: float((x > 0).mean())),
            "avg_pnl_pct": g["pnl_pct"].mean(),
        }).sort_values("total_pnl", ascending=False)

    return PortfolioResult(equity_curve=eq, weights=weights, trades=trades_df,
                           metrics=metrics, contribution=contrib)


def format_portfolio(res: PortfolioResult, top: int = 10) -> str:
    """格式化组合回测结果。"""
    m = res.metrics
    lines = [f"\n组合回测（{m.get('n_symbols', 0)} 只候选，{m.get('num_bars', 0)} 根 bar）\n"]
    lines.append(f"  {res.summary()}")
    lines.append(
        f"  Sortino {m.get('sortino', 0):+.2f}   Calmar {m.get('calmar', 0):+.2f}   "
        f"年化波动 {m.get('volatility', 0):.2%}   平均现金 {m.get('cash_pct_mean', 0):.1%}   "
        f"最多同时持 {m.get('max_positions_held', 0)} 只"
    )
    if not res.contribution.empty:
        lines.append(f"\n  逐标的贡献（前 {top}）：")
        lines.append(f"  {'标的':<14}{'交易':>6}{'总盈亏':>14}{'胜率':>8}{'平均每笔':>10}")
        lines.append("  " + "-" * 54)
        for sym, r in res.contribution.head(top).iterrows():
            lines.append(f"  {sym:<14}{int(r['trades']):>6}{r['total_pnl']:>+14,.0f}"
                         f"{r['win_rate']:>7.0%}{r['avg_pnl_pct']:>+10.2%}")
    return "\n".join(lines) + "\n"


def compare_allocations(
    bars: dict[str, pd.DataFrame],
    strategy: SignalFunc,
    cfg: PortfolioConfig | None = None,
) -> pd.DataFrame:
    """对比三种资金分配方式，回答「这笔钱该怎么分」。"""
    import dataclasses

    base = cfg or PortfolioConfig()
    rows = []
    for alloc in ("equal", "inverse_vol", "score"):
        c = dataclasses.replace(base, allocation=alloc)
        try:
            r = run_portfolio(bars, strategy, c)
        except Exception as e:
            logger.debug("分配方式 %s 失败：%s", alloc, e)
            continue
        m = r.metrics
        rows.append({
            "分配方式": {"equal": "等权", "inverse_vol": "波动率反比",
                         "score": "信号强弱"}[alloc],
            "总收益": m["total_return"], "年化": m["annual_return"],
            "夏普": m["sharpe"], "最大回撤": m["max_drawdown"],
            "年化波动": m["volatility"], "换手x/年": m["annual_turnover"],
            "平均持仓": m["avg_positions"],
        })
    return pd.DataFrame(rows)
