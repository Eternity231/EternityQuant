"""CLI 入口（typer）。命令骨架见 problem 5 决议：

    eq watch <symbol>                       # 查个股快照
    eq scan <market> [--by change|volume]   # 扫市场
    eq monitor add/list                     # 盯（待）
    eq portfolio add/show                   # 持仓（待）
    eq research <symbol>                    # 研（待）
    eq dash                                 # Streamlit 仪表盘（待）

第一版实现 `eq watch` + `eq scan`（仅 A 股）。
"""

from __future__ import annotations

import typer

from eq.backtest import BacktestConfig, EventDrivenBacktester, VectorizedBacktester
from eq.core import monitor as mon_svc
from eq.core import portfolio as pf_svc
from eq.core import scheduler as sched_svc
from eq.core import watchlist as wl_svc
from eq.core.notifier import available_channels
from eq.core.scanner import Market, SortBy, format_scan, scan as market_scan
from eq.core.watcher import format_snapshot
from eq.strategy.factors import ml as ml_svc
from eq.strategy.signals import adx_trend, bollinger_break, ema_cross, rsi_reversal

app = typer.Typer(
    name="eq",
    help="EternityQuant — 个人散户量化助手",
    no_args_is_help=True,
    add_completion=False,
)

# 子命令组：eq watchlist ...
watchlist_app = typer.Typer(help="自选股管理（增删查）", no_args_is_help=True)
app.add_typer(watchlist_app, name="watchlist")

# 子命令组：eq portfolio ...
portfolio_app = typer.Typer(help="持仓管理（建仓/加仓/减仓/清仓/止损止盈）", no_args_is_help=True)
app.add_typer(portfolio_app, name="portfolio")

# 子命令组：eq monitor ...
monitor_app = typer.Typer(help="监控规则（注册/启停/扫描触发）", no_args_is_help=True)
app.add_typer(monitor_app, name="monitor")

# 子命令组：eq ml ...
ml_app = typer.Typer(help="qlib ML 模型管理（注册/激活/列表/预测）", no_args_is_help=True)
app.add_typer(ml_app, name="ml")

# 子命令组：eq scheduler ...
scheduler_app = typer.Typer(help="定时推送服务（cron 表达式 + APScheduler）", no_args_is_help=True)
app.add_typer(scheduler_app, name="scheduler")


@app.command(help="看个股行情快照（最近一根日线 + 涨跌幅）")
def watch(
    symbol: str = typer.Argument(help="股票符号，如 600519.SH、AAPL.US、00700.HK"),
):
    try:
        typer.echo(format_snapshot(symbol))
    except Exception as e:
        typer.echo(f"拉取失败：{e}", err=True)
        raise typer.Exit(1)


@app.command(help="扫全市场，按指定字段排序展示前 N 名")
def scan(
    market: Market = typer.Argument("A", help="市场：A=沪深京，HK=港股，US=美股，CRYPTO=加密"),
    sort_by: SortBy = typer.Option("change_pct", "--by", "-b", help="排序键：change_pct|volume|amount"),
    top_n: int = typer.Option(30, "--top", "-n", help="前 N 名"),
):
    try:
        df = market_scan(market, sort_by=sort_by, top_n=top_n)
        if df.empty:
            typer.echo(f"{market} 扫描结果为空")
            raise typer.Exit(0)
        typer.echo(format_scan(df, sort_by, market=market))
    except Exception as e:
        typer.echo(f"扫描失败：{e}", err=True)
        raise typer.Exit(1)


@app.command("research", help="个股深度研究（按市场自动选数据源汇总基本面/资金/新闻/研报等）")
def research(
    symbol: str = typer.Argument(help="股票符号，如 600519.SH、AAPL.US、00700.HK"),
    sections: str = typer.Option("", "--sections", "-s", help="指定板块，逗号分隔，如 financial,news；缺省按市场全拉"),
):
    from eq.core.research import format_research, research as do_research
    secs = [s.strip() for s in sections.split(",") if s.strip()] or None
    try:
        report = do_research(symbol, sections=secs)
        typer.echo(format_research(report))
    except Exception as e:
        typer.echo(f"研究失败：{e}", err=True)
        raise typer.Exit(1)


# ---------- eq watchlist 子命令 ----------

@watchlist_app.command("add", help="加入自选股")
def wl_add(
    symbol: str = typer.Argument(help="股票符号，如 600519.SH"),
    reason: str = typer.Option("", "--reason", "-r", help="加入理由"),
    tags: str = typer.Option("", "--tags", "-t", help="标签，逗号分隔，如 白酒,龙头"),
):
    rowid = wl_svc.add(symbol, reason=reason, tags=tags)
    if rowid == 0:
        typer.echo(f"{symbol} 已在自选列表")
    else:
        typer.echo(f"已加入自选：{symbol}")


@watchlist_app.command("remove", help="移出自选股")
def wl_remove(
    symbol: str = typer.Argument(help="股票符号"),
):
    if wl_svc.remove(symbol):
        typer.echo(f"已移出自选：{symbol}")
    else:
        typer.echo(f"{symbol} 不在自选列表", err=True)
        raise typer.Exit(1)


@watchlist_app.command("list", help="列出全部自选股")
def wl_list():
    rows = wl_svc.list_all()
    if not rows:
        typer.echo("自选列表为空")
        return
    print(f"\n自选列表（共 {len(rows)} 只）：\n")
    print(f"{'符号':<14} {'市场':<6} {'名称':<10} {'加入时间':<20} {'标签':<10} {'理由'}")
    print("-" * 90)
    for r in rows:
        name = r["name"] or "-"
        added = str(r["added_at"] or "-")
        tags = r["tags"] or "-"
        reason = r["reason"] or "-"
        market = r["market"] or "-"
        print(f"{r['symbol']:<14} {market:<6} {name:<10} {added:<20} {tags:<10} {reason}")


@watchlist_app.command("find", help="查单只是否在自选")
def wl_find(
    symbol: str = typer.Argument(help="股票符号"),
):
    r = wl_svc.find(symbol)
    if r is None:
        typer.echo(f"{symbol} 不在自选")
        raise typer.Exit(1)
    typer.echo(
        f"{r['symbol']}  市场={r['market']}  名称={r['name'] or '-'}  "
        f"加入={r['added_at']}  标签={r['tags'] or '-'}  理由={r['reason'] or '-'}"
    )


# ---------- eq portfolio 子命令 ----------

@portfolio_app.command("buy", help="建仓（若已持仓自动转加仓）")
def pf_buy(
    symbol: str = typer.Argument(help="股票符号"),
    shares: float = typer.Argument(help="股数"),
    price: float = typer.Argument(help="成交价"),
    stop_loss: float = typer.Option(None, "--stop", "-s", help="止损价"),
    take_profit: float = typer.Option(None, "--target", "-t", help="止盈价"),
    note: str = typer.Option("", "--note", "-n", help="备注"),
):
    pos_id = pf_svc.open_position(symbol, shares, price, stop_loss=stop_loss, take_profit=take_profit, note=note)
    typer.echo(f"持仓 #{pos_id} 已记录：{symbol}  {shares} 股 @ {price}")


@portfolio_app.command("add", help="加仓（加权平均更新成本价）")
def pf_add(
    symbol: str = typer.Argument(help="股票符号"),
    shares: float = typer.Argument(help="加仓股数"),
    price: float = typer.Argument(help="成交价"),
    note: str = typer.Option("", "--note", "-n", help="备注"),
):
    try:
        pos_id = pf_svc.add(symbol, shares, price, note=note)
        typer.echo(f"已加仓：{symbol} +{shares} 股 @ {price}")
    except ValueError as e:
        typer.echo(f"加仓失败：{e}", err=True)
        raise typer.Exit(1)


@portfolio_app.command("trim", help="减仓（不动成本价，累加已实现盈亏，减到 0 自动清仓）")
def pf_trim(
    symbol: str = typer.Argument(help="股票符号"),
    shares: float = typer.Argument(help="减仓股数"),
    price: float = typer.Argument(help="成交价"),
    note: str = typer.Option("", "--note", "-n", help="备注"),
):
    try:
        pf_svc.trim(symbol, shares, price, note=note)
        typer.echo(f"已减仓：{symbol} -{shares} 股 @ {price}")
    except ValueError as e:
        typer.echo(f"减仓失败：{e}", err=True)
        raise typer.Exit(1)


@portfolio_app.command("sell", help="清仓（全部卖出并结算盈亏）")
def pf_sell(
    symbol: str = typer.Argument(help="股票符号"),
    price: float = typer.Argument(help="成交价"),
    note: str = typer.Option("", "--note", "-n", help="备注"),
):
    pos = pf_svc.get_open(symbol)
    if pos is None:
        typer.echo(f"无 open 持仓：{symbol}", err=True)
        raise typer.Exit(1)
    try:
        pf_svc.trim(symbol, pos["shares"], price, note=note or "清仓")
        typer.echo(f"已清仓：{symbol} @ {price}")
    except ValueError as e:
        typer.echo(f"清仓失败：{e}", err=True)
        raise typer.Exit(1)


@portfolio_app.command("stops", help="设置止损/止盈价")
def pf_stops(
    symbol: str = typer.Argument(help="股票符号"),
    stop_loss: float = typer.Option(None, "--stop", "-s", help="止损价"),
    take_profit: float = typer.Option(None, "--target", "-t", help="止盈价"),
):
    if not pf_svc.set_stops(symbol, stop_loss=stop_loss, take_profit=take_profit):
        typer.echo(f"未更新（{symbol} 无 open 持仓或未提供任何价格）", err=True)
        raise typer.Exit(1)
    typer.echo(f"已更新止损止盈：{symbol}  stop={stop_loss or '-'}  target={take_profit or '-'}")


@portfolio_app.command("list", help="列出当前持仓")
def pf_list():
    rows = pf_svc.list_open()
    if not rows:
        typer.echo("当前无持仓")
        return
    print(f"\n当前持仓（共 {len(rows)} 只）：\n")
    print(f"{'符号':<14} {'市场':<6} {'股数':>10} {'成本':>10} {'止损':>10} {'止盈':>10} {'已实现':>12} {'建仓时间'}")
    print("-" * 110)
    for r in rows:
        print(
            f"{r['symbol']:<14} {r['market'] or '-':<6} {r['shares']:>10.0f} {r['cost_price']:>10.2f} "
            f"{(r['stop_loss'] or 0):>10.2f} {(r['take_profit'] or 0):>10.2f} "
            f"{r['realized_pnl']:>+12.2f} {r['opened_at']}"
        )


@portfolio_app.command("history", help="查某只股票的交易历史")
def pf_history(
    symbol: str = typer.Argument(help="股票符号"),
    limit: int = typer.Option(50, "--limit", "-n", help="最近 N 条"),
):
    rows = pf_svc.trade_history(symbol, limit=limit)
    if not rows:
        typer.echo(f"{symbol} 无交易历史")
        return
    print(f"\n{symbol} 交易历史（最近 {len(rows)} 条）：\n")
    print(f"{'时间':<22} {'动作':<6} {'股数':>10} {'价格':>10} {'备注'}")
    print("-" * 70)
    for r in rows:
        print(f"{str(r['executed_at']):<22} {r['action']:<6} {r['shares']:>10.0f} {r['price']:>10.2f} {r['note'] or '-'}")


@portfolio_app.command("closed", help="列出最近已清仓的持仓")
def pf_closed(
    limit: int = typer.Option(20, "--limit", "-n", help="最近 N 条"),
):
    rows = pf_svc.list_closed(limit=limit)
    if not rows:
        typer.echo("无已清仓记录")
        return
    print(f"\n已清仓记录（最近 {len(rows)} 条）：\n")
    print(f"{'符号':<14} {'市场':<6} {'成本':>10} {'已实现盈亏':>14} {'建仓':<22} {'清仓':<22}")
    print("-" * 90)
    for r in rows:
        print(
            f"{r['symbol']:<14} {r['market'] or '-':<6} {r['cost_price']:>10.2f} "
            f"{r['realized_pnl']:>+14.2f} {str(r['opened_at']):<22} {str(r['closed_at']):<22}"
        )


# ---------- eq monitor 子命令 ----------

@monitor_app.command("add", help="注册监控规则")
def mon_add(
    symbol: str = typer.Argument(None, help="股票符号；省略表示全市场规则"),
    rule_type: str = typer.Argument(help=f"规则类型，可选：{','.join(sorted(mon_svc.RULE_TYPES))}"),
    params_json: str = typer.Argument("{}", help="参数 JSON，如 '{\"level\":1800,\"direction\":\"up\"}'"),
    channels: str = typer.Option("desktop", "--channels", "-c", help="推送通道，逗号分隔，如 desktop,wechat_work"),
):
    import json as _json
    try:
        params = _json.loads(params_json)
    except _json.JSONDecodeError as e:
        typer.echo(f"params JSON 解析失败：{e}", err=True)
        raise typer.Exit(1)
    try:
        rid = mon_svc.add_rule(symbol or None, rule_type, params, channels=[c.strip() for c in channels.split(",") if c.strip()])
    except ValueError as e:
        typer.echo(f"注册失败：{e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"已注册规则 #{rid}：{symbol or '全市场'}  {rule_type}  {params}  通道={channels}")


@monitor_app.command("remove", help="删除监控规则")
def mon_remove(
    rule_id: int = typer.Argument(help="规则 id"),
):
    if mon_svc.remove_rule(rule_id):
        typer.echo(f"已删除规则 #{rule_id}")
    else:
        typer.echo(f"规则 #{rule_id} 不存在", err=True)
        raise typer.Exit(1)


@monitor_app.command("list", help="列出所有规则")
def mon_list(
    enabled_only: bool = typer.Option(False, "--enabled", "-e", help="仅看 enabled"),
):
    rules = mon_svc.list_rules(enabled_only=enabled_only)
    if not rules:
        typer.echo("无监控规则")
        return
    print(f"\n监控规则（共 {len(rules)} 条）：\n")
    print(f"{'#':<5} {'符号':<14} {'类型':<14} {'参数':<30} {'通道':<20} {'启':<4} {'触发':<6} {'上次触发'}")
    print("-" * 130)
    for r in rules:
        sym = r["symbol"] or "全市场"
        params = str(r["params"])[:28]
        channels = ",".join(r["channels"])[:18]
        last = str(r["last_fired_at"] or "-")[:19]
        print(f"{r['id']:<5} {sym:<14} {r['type']:<14} {params:<30} {channels:<20} {'是' if r['enabled'] else '否':<4} {r['fire_count']:<6} {last}")


@monitor_app.command("enable", help="启用规则")
def mon_enable(rule_id: int = typer.Argument(help="规则 id")):
    if mon_svc.set_enabled(rule_id, True):
        typer.echo(f"已启用规则 #{rule_id}")
    else:
        typer.echo(f"规则 #{rule_id} 不存在", err=True)
        raise typer.Exit(1)


@monitor_app.command("disable", help="停用规则")
def mon_disable(rule_id: int = typer.Argument(help="规则 id")):
    if mon_svc.set_enabled(rule_id, False):
        typer.echo(f"已停用规则 #{rule_id}")
    else:
        typer.echo(f"规则 #{rule_id} 不存在", err=True)
        raise typer.Exit(1)


@monitor_app.command("run", help="立即扫描所有 enabled 规则并触发推送")
def mon_run():
    fired = mon_svc.run_all()
    typer.echo(f"扫描完成，触发 {fired} 条规则")


@monitor_app.command("channels", help="列出当前可用的推送通道")
def mon_channels():
    chs = available_channels()
    if not chs:
        typer.echo("无可用通道（请配置 ~/.eternityquant/.env）")
        return
    typer.echo("可用推送通道：" + ", ".join(chs))


# ---------- eq backtest 命令 ----------

_BUILTIN_STRATEGIES = {
    "ema_cross": ema_cross,
    "adx_trend": adx_trend,
    "rsi_reversal": rsi_reversal,
    "bollinger_break": bollinger_break,
}


@app.command("backtest", help="回测内置策略（双引擎可选，自动外存 parquet）")
def backtest(
    symbol: str = typer.Argument(help="股票符号，如 600519.SH"),
    strategy: str = typer.Argument(help=f"策略名，可选：{','.join(sorted(_BUILTIN_STRATEGIES))}"),
    engine: str = typer.Option("vectorized", "--engine", "-e", help="引擎：vectorized（快）| event_driven（准）"),
    days: int = typer.Option(365, "--days", "-d", help="回测窗口天数"),
    initial_cash: float = typer.Option(1_000_000, "--cash", "-c", help="初始现金"),
    commission_bps: float = typer.Option(2.5, "--commission", help="单边手续费（万分之）"),
    slippage_bps: float = typer.Option(5.0, "--slippage", help="单边滑点（万分之）"),
    save: bool = typer.Option(True, "--save/--no-save", help="是否外存 parquet + 入 backtest_runs 表"),
):
    if strategy not in _BUILTIN_STRATEGIES:
        typer.echo(f"未知策略 {strategy}，可选：{','.join(sorted(_BUILTIN_STRATEGIES))}", err=True)
        raise typer.Exit(1)
    if engine not in ("vectorized", "event_driven"):
        typer.echo(f"未知引擎 {engine}，可选：vectorized / event_driven", err=True)
        raise typer.Exit(1)
    from eq.data.market import get_recent_bars
    try:
        df = get_recent_bars(symbol, days=days)
    except Exception as e:
        typer.echo(f"拉行情失败：{e}", err=True)
        raise typer.Exit(1)
    if df.empty:
        typer.echo("行情为空，无法回测", err=True)
        raise typer.Exit(1)
    cfg = BacktestConfig(
        initial_cash=initial_cash,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        engine=engine,
    )
    backtester = VectorizedBacktester() if engine == "vectorized" else EventDrivenBacktester()
    result = backtester.run(df, _BUILTIN_STRATEGIES[strategy], cfg)
    typer.echo(f"\n回测 {symbol} 用 {strategy}（{engine}，{len(df)} 日）：")
    typer.echo(f"  {result.summary()}")
    if not result.trades.empty:
        typer.echo(f"\n交易明细（前 5 笔）：")
        print(result.trades.head(5).to_string(index=False))
    if save:
        from eq.backtest.store import save_result
        run_id = save_result(result, symbol=symbol, strategy_name=strategy)
        typer.echo(f"\n已外存：run_id={run_id}（用 `eq bt show {run_id}` 查完整结果）")


# 子命令组：eq bt ... （回测历史管理，避免破坏 eq backtest 主命令）
bt_app = typer.Typer(help="回测历史管理（list/show/remove）", no_args_is_help=True)
app.add_typer(bt_app, name="bt")


@bt_app.command("list", help="列出最近回测记录")
def bt_list(
    symbol: str = typer.Option(None, "--symbol", "-s", help="按标的过滤"),
    limit: int = typer.Option(20, "--limit", "-n", help="最近 N 条"),
):
    from eq.backtest.store import list_runs
    runs = list_runs(symbol=symbol, limit=limit)
    if not runs:
        typer.echo("无回测记录")
        return
    print(f"\n回测记录（最近 {len(runs)} 条）：\n")
    print(f"{'run_id':<24} {'标的':<14} {'策略':<16} {'引擎':<14} {'总收益':>10} {'夏普':>8} {'时间'}")
    print("-" * 110)
    for r in runs:
        m = r["metrics"]
        print(
            f"{r['id']:<24} {r['symbol']:<14} {r['strategy_name']:<16} {r['engine']:<14} "
            f"{m.get('total_return', 0):>+9.2%} {m.get('sharpe', 0):>+8.2f} {str(r['created_at'])[:19]}"
        )


@bt_app.command("show", help="查某次回测的完整结果（metadata + 权益曲线 + 交易明细）")
def bt_show(
    run_id: str = typer.Argument(help="run_id"),
    details: bool = typer.Option(False, "--details", "-d", help="显示权益曲线和交易明细完整数据"),
):
    from eq.backtest.store import load_result
    try:
        bundle = load_result(run_id)
    except KeyError as e:
        typer.echo(f"{e}", err=True)
        raise typer.Exit(1)
    meta = bundle["meta"]
    m = meta["metrics"]
    typer.echo(f"\n回测 {meta['symbol']} 用 {meta['strategy_name']}（{meta['engine']}）@ {meta['created_at']}")
    typer.echo(f"  总收益 {m.get('total_return', 0):+.2%}  年化 {m.get('annual_return', 0):+.2%}  夏普 {m.get('sharpe', 0):+.2f}  最大回撤 {m.get('max_drawdown', 0):+.2%}  胜率 {m.get('win_rate', 0):.1%}  交易 {m.get('num_trades', 0)} 笔")
    if details:
        typer.echo(f"\n权益曲线（前 5 日）：")
        print(bundle["equity"].head(5).to_string())
        if not bundle["trades"].empty:
            typer.echo(f"\n交易明细（前 5 笔）：")
            print(bundle["trades"].head(5).to_string(index=False))


@bt_app.command("remove", help="删除某次回测记录（SQLite metadata + parquet 文件）")
def bt_remove(run_id: str = typer.Argument(help="run_id")):
    from eq.backtest.store import remove_run
    if remove_run(run_id):
        typer.echo(f"已删除回测 {run_id}")
    else:
        typer.echo(f"回测 {run_id} 不存在", err=True)
        raise typer.Exit(1)


# 子命令组：eq ml ...
ml_app = typer.Typer(help="qlib ML 模型管理（注册/激活/列表/预测）", no_args_is_help=True)
app.add_typer(ml_app, name="ml")


@ml_app.command("register", help="登记一个训练完成的模型")
def ml_register(
    name: str = typer.Argument(help="模型名，如 a-share_lightgbm_v1"),
    universe: str = typer.Argument(help="标的池，如 csi300"),
    algo: str = typer.Argument("lightgbm", help="算法：lightgbm/xgboost/linear/mlp"),
    horizon: int = typer.Argument(5, help="预测窗口（天）"),
    train_period: str = typer.Argument("2020-01-01~2025-12-31", help="训练区间"),
    features_json: str = typer.Option("[]", "--features", "-f", help="特征 JSON 列表"),
    model_path: str = typer.Option("", "--path", "-p", help="模型文件路径"),
    notes: str = typer.Option("", "--note", "-n", help="备注"),
):
    import json as _json
    try:
        feats = _json.loads(features_json)
    except _json.JSONDecodeError as e:
        typer.echo(f"features JSON 解析失败：{e}", err=True)
        raise typer.Exit(1)
    mid = ml_svc.register_model(
        name=name, universe=universe, features=feats, algo=algo, horizon=horizon,
        train_period=train_period, model_path=model_path, notes=notes,
    )
    typer.echo(f"已登记模型 {mid}（{name}，universe={universe}，algo={algo}，horizon={horizon}）")


@ml_app.command("activate", help="激活某模型（同 universe 其他自动停用）")
def ml_activate(model_id: str = typer.Argument(help="模型 id")):
    if ml_svc.activate(model_id):
        typer.echo(f"已激活模型 {model_id}")
    else:
        typer.echo(f"模型 {model_id} 不存在", err=True)
        raise typer.Exit(1)


@ml_app.command("list", help="列出模型")
def ml_list(
    universe: str = typer.Option(None, "--universe", "-u", help="按池过滤"),
):
    rows = ml_svc.list_models(universe=universe)
    if not rows:
        typer.echo("无模型记录")
        return
    print(f"\nML 模型（共 {len(rows)} 个）：\n")
    print(f"{'id':<22} {'name':<24} {'universe':<10} {'algo':<10} {'horizon':>7} {'激':<3} {'训练时间'}")
    print("-" * 110)
    for r in rows:
        active = "是" if r["is_active"] else "否"
        print(
            f"{r['id']:<22} {(r['name'] or '-')[:22]:<24} {r['universe'] or '-':<10} "
            f"{r['algo']:<10} {r['horizon']:>7} {active:<3} {str(r['trained_at'])[:19]}"
        )


@ml_app.command("predict", help="对某标的写入一条预测分数（手工录入，用于测试或补漏）")
def ml_predict(
    model_id: str = typer.Argument(help="模型 id"),
    symbol: str = typer.Argument(help="股票符号"),
    score: float = typer.Argument(help="预测分数"),
    date: str = typer.Option("", "--date", "-d", help="YYYY-MM-DD，默认今天"),
):
    import datetime as dt
    d = dt.date.fromisoformat(date) if date else dt.date.today()
    ml_svc.save_prediction(model_id, symbol, d, score)
    typer.echo(f"已写入预测：{model_id} / {symbol} / {d} / score={score}")


@ml_app.command("train", help="走 qlib workflow 真训练（Alpha158 + LightGBM/PyTorch，可选 GPU/CUDA）")
def ml_train(
    universe: str = typer.Argument("csi300", help="标的池，如 csi300/csi500"),
    horizon: int = typer.Argument(5, help="预测窗口（天）"),
    algo: str = typer.Option("lightgbm", "--algo", "-a", help="lightgbm | alstm | gru | lstm | mlp"),
    train_start: str = typer.Option("2015-01-01", "--train-start", help="训练区间起"),
    train_end: str = typer.Option("2020-08-31", "--train-end", help="训练区间止"),
    valid_start: str = typer.Option("2020-09-01", "--valid-start", help="验证区间起"),
    valid_end: str = typer.Option("2020-09-25", "--valid-end", help="验证区间止（qlib 数据末日）"),
    device: str = typer.Option("cpu", "--device", "-d", help="cpu | gpu | cuda（LightGBM gpu=OpenCL；PyTorch cuda=真CUDA，3060主场）"),
    name: str = typer.Option("", "--name", "-n", help="模型名，默认自动生成"),
):
    from eq.strategy.factors.ml_workflow import train as wf_train, train_torch as wf_train_torch, _TORCH_ALGOS
    try:
        if algo in _TORCH_ALGOS:
            # PyTorch 模型默认 cuda（GPU 参数透传给 qlib，cuda → GPU=0）
            result = wf_train_torch(
                universe=universe, horizon=horizon, algo=algo,
                train_start=train_start, train_end=train_end,
                valid_start=valid_start, valid_end=valid_end,
                device=device, name=name or None,
            )
        else:
            result = wf_train(
                universe=universe, horizon=horizon, algo=algo,
                train_start=train_start, train_end=train_end,
                valid_start=valid_start, valid_end=valid_end,
                device=device, name=name or None,
            )
    except Exception as e:
        typer.echo(f"训练失败：{e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"\n训练完成：model_id={result['model_id']}")
    typer.echo(f"  IC={result['metrics']['ic']:+.4f}  algo={algo}  device={device}  模型文件={result['model_path']}")
    typer.echo(f"  用 `eq ml activate {result['model_id']}` 激活，再 `eq ml predict-batch` 批量预测")


@ml_app.command("predict-batch", help="用激活模型批量预测全 universe，写入 ml_predictions 表")
def ml_predict_batch(
    model_id: str = typer.Argument(help="模型 id"),
    predict_date: str = typer.Option("", "--date", "-d", help="YYYY-MM-DD，默认 qlib 数据末日 2020-09-25"),
    top_n: int = typer.Option(50, "--top", "-n", help="前 N 名"),
):
    from eq.strategy.factors.ml_workflow import predict_batch
    try:
        df = predict_batch(model_id, predict_date=predict_date or None, top_n=top_n)
    except Exception as e:
        typer.echo(f"预测失败：{e}", err=True)
        raise typer.Exit(1)
    if df.empty:
        typer.echo("预测结果为空")
        return
    print(f"\n预测前 {len(df)} 名（已写入 ml_predictions 表）：\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    app()


# ---------- eq scheduler 子命令 ----------

@scheduler_app.command("add", help="注册定时任务（cron 表达式）")
def sched_add(
    name: str = typer.Argument(help="任务名，唯一"),
    cron_expr: str = typer.Argument(help="cron 表达式（分 时 日 月 周），如 '0 16 * * 1-5' = 工作日 16:00"),
    action: str = typer.Argument(help=f"动作，可选：{','.join(sorted(sched_svc._ACTIONS))}"),
    channels: str = typer.Option("desktop", "--channels", "-c", help="推送通道，逗号分隔"),
    params_json: str = typer.Option("{}", "--params", "-p", help="参数 JSON，如 '{\"market\":\"A\",\"top_n\":20}'"),
):
    import json as _json
    try:
        params = _json.loads(params_json)
    except _json.JSONDecodeError as e:
        typer.echo(f"params JSON 解析失败：{e}", err=True)
        raise typer.Exit(1)
    try:
        jid = sched_svc.add_job(
            name=name, cron_expr=cron_expr, action=action,
            params=params, channels=[c.strip() for c in channels.split(",") if c.strip()],
        )
    except ValueError as e:
        typer.echo(f"注册失败：{e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"已注册任务 #{jid}：{name}  {cron_expr}  {action}  {params}")


@scheduler_app.command("remove", help="删除定时任务")
def sched_remove(job_id: int = typer.Argument(help="任务 id")):
    if sched_svc.remove_job(job_id):
        typer.echo(f"已删除任务 #{job_id}")
    else:
        typer.echo(f"任务 #{job_id} 不存在", err=True)
        raise typer.Exit(1)


@scheduler_app.command("list", help="列出所有定时任务")
def sched_list(enabled_only: bool = typer.Option(False, "--enabled", "-e", help="仅看 enabled")):
    jobs = sched_svc.list_jobs(enabled_only=enabled_only)
    if not jobs:
        typer.echo("无定时任务")
        return
    print(f"\n定时任务（共 {len(jobs)} 个）：\n")
    print(f"{'#':<4} {'名称':<20} {'cron':<18} {'动作':<14} {'启':<3} {'次数':<6} {'上次状态':<8} {'上次运行'}")
    print("-" * 110)
    for j in jobs:
        last = str(j["last_run_at"] or "-")[:19]
        status = j["last_run_status"] or "-"
        enabled = "是" if j["enabled"] else "否"
        print(f"{j['id']:<4} {j['name'][:18]:<20} {j['cron_expr']:<18} {j['action']:<14} {enabled:<3} {j['run_count']:<6} {status:<8} {last}")


@scheduler_app.command("enable", help="启用定时任务")
def sched_enable(job_id: int = typer.Argument(help="任务 id")):
    if sched_svc.set_enabled(job_id, True):
        typer.echo(f"已启用任务 #{job_id}")
    else:
        typer.echo(f"任务 #{job_id} 不存在", err=True)
        raise typer.Exit(1)


@scheduler_app.command("disable", help="停用定时任务")
def sched_disable(job_id: int = typer.Argument(help="任务 id")):
    if sched_svc.set_enabled(job_id, False):
        typer.echo(f"已停用任务 #{job_id}")
    else:
        typer.echo(f"任务 #{job_id} 不存在", err=True)
        raise typer.Exit(1)


@scheduler_app.command("run", help="立即执行某任务一次（不等触发）")
def sched_run(job_id: int = typer.Argument(help="任务 id")):
    sched = sched_svc.get_scheduler()
    sched.start()
    if not sched.run_now(job_id):
        typer.echo(f"任务 #{job_id} 不存在", err=True)
        raise typer.Exit(1)
    typer.echo(f"任务 #{job_id} 已触发，查看状态用 eq scheduler list")


@scheduler_app.command("daemon", help="启动调度器常驻进程（按 cron 定时执行任务）")
def sched_daemon():
    sched = sched_svc.get_scheduler()
    sched.start()
    typer.echo("调度器已启动，Ctrl+C 退出")
    try:
        import time
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        typer.echo("\n退出中...")
        sched.stop()


# ---------- eq dash 命令（放末尾，避免 streamlit 启动逻辑被其他装饰器干扰） ----------

@app.command("dash", help="启动 Streamlit 仪表盘（本地网页看板）")
def dash(
    port: int = typer.Option(8501, "--port", "-p", help="本地端口"),
):
    from eq.web import run_dashboard
    code = run_dashboard(port=port)
    if code != 0:
        typer.echo(f"Streamlit 异常退出：{code}", err=True)
        raise typer.Exit(code)
