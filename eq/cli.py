"""CLI 入口（typer）。命令骨架见 problem 5 决议：

    eq watch <symbol>                       # 查个股快照
    eq scan <market> [--by change|volume]   # 扫市场
    eq monitor add/list                     # 盯（待）
    eq portfolio add/show                   # 持仓（待）
    eq dash                                 # Streamlit 仪表盘（待）

第一版实现 `eq watch` + `eq scan`（仅 A 股）。
"""

from __future__ import annotations

# Windows 控制台默认 cp936/GBK，print() 中文会 mojibake（港股5m完成 → 赟5m姝）
# 强制 stdout/stderr UTF-8，保证中文输出在任何 Windows 控制台都不乱码
import sys as _sys
try:
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass  # 某些嵌入环境无 reconfigure（如 streamlit subprocess），静默跳过

# torch DLL 预热（Windows + cu132 坑：qlib 集成链触发 torch 延迟加载 c10.dll 失败）
# 放在所有 eq.* import 之前，否则 eq.strategy.factors.ml 会先拖 qlib 链触发 torch 延迟加载
# 仅 cuda 可用时才 init，避免无 GPU 机器每次 CLI 都拉 CUDA driver
try:
    import torch as _torch
    if _torch.cuda.is_available():
        _torch.cuda.init()
except (ImportError, OSError):
    pass

import numpy as np
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
# （注意：只能建一次。此前文件中段又建了一个同名 ml_app 并二次 add_typer，
#   两个同名 group 注册进同一个 app，行为取决于 click 的解析顺序。）
ml_app = typer.Typer(help="qlib ML 模型管理（注册/激活/列表/预测）", no_args_is_help=True)
app.add_typer(ml_app, name="ml")

# 子命令组：eq scheduler ...
scheduler_app = typer.Typer(help="定时推送服务（cron 表达式 + APScheduler）", no_args_is_help=True)
app.add_typer(scheduler_app, name="scheduler")

hk_app = typer.Typer(help="港股数据管道（Sina 源）", no_args_is_help=True)
app.add_typer(hk_app, name="hk")

# 子命令组：eq data ...
data_app = typer.Typer(help="数据收集（A股/港股日线/分钟线/美股）", no_args_is_help=True)
app.add_typer(data_app, name="data")


@app.command(help="看个股行情快照（最近一根日线 + 涨跌幅）")
def watch(
    symbol: str = typer.Argument(help="股票符号，如 600519.SH、AAPL.US、00700.HK"),
    realtime: bool = typer.Option(False, "--realtime", "-r",
                                  help="走实时行情源（新浪/腾讯），盘中返回当前价而非上一交易日收盘"),
    source: str = typer.Option("", "--source", "-S",
                               help="指定优先数据源，逗号分隔，如 tencent,sina（见 eq data sources）"),
):
    prefer = [x.strip() for x in source.split(",") if x.strip()] or None
    try:
        typer.echo(format_snapshot(symbol, realtime=realtime, prefer=prefer))
    except Exception as e:
        typer.echo(f"拉取失败：{e}", err=True)
        raise typer.Exit(1) from e


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
        raise typer.Exit(1) from e


def _resolve_symbols(source: str, top_n: int = 100) -> tuple[list[str], str]:
    """把 ``--from`` 的值解析成标的列表。

    支持 ``watchlist`` / ``portfolio`` / ``both`` / 市场代码（A/HK/US/CRYPTO）
    / 逗号分隔的代码列表。screen、cache warm、bt robust 三处都要这套逻辑，
    抽出来避免各写一遍走样。

    Returns:
        ``(symbols, 人话标签)``
    """
    src = str(source).strip()
    if src in ("watchlist", "portfolio", "both"):
        syms: list[str] = []
        if src in ("watchlist", "both"):
            syms += [r["symbol"] for r in wl_svc.list_all()]
        if src in ("portfolio", "both"):
            syms += [r["symbol"] for r in pf_svc.list_open()]
        label = {"watchlist": "自选股", "portfolio": "当前持仓", "both": "自选+持仓"}[src]
        return list(dict.fromkeys(syms)), label
    if src.upper() in ("A", "HK", "US", "CRYPTO"):
        df = market_scan(src.upper(), sort_by="amount", top_n=top_n)
        syms = df["symbol"].astype(str).tolist()
        return syms, f"{src.upper()} 市场成交额前 {len(syms)}"
    syms = list(dict.fromkeys(x.strip() for x in src.split(",") if x.strip()))
    return syms, "指定代码"


@app.command("screen", help="技术选股：对自选/持仓/市场榜跑技术条件筛选（RSI/金叉/放量/突破等）")
def screen(
    conditions: str = typer.Argument(
        help="筛选条件，逗号分隔。可选：rsi_oversold,rsi_overbought,golden_cross,death_cross,"
             "macd_golden,macd_death,above_ma,below_ma,volume_spike,near_high,near_low,"
             "breakout,pullback,squeeze",
    ),
    source: str = typer.Option("watchlist", "--from", "-f",
                               help="标的来源：watchlist（自选）| portfolio（持仓）| A/HK/US/CRYPTO（市场榜）| 逗号分隔的代码"),
    mode: str = typer.Option("all", "--mode", "-m", help="all=全部条件满足 | any=任一满足"),
    top_n: int = typer.Option(100, "--top", "-n", help="市场榜来源时取前 N 只作为候选池"),
    days: int = typer.Option(120, "--days", "-d", help="每只拉多少根日线（部分条件需 60+）"),
    params_json: str = typer.Option("{}", "--params", "-p",
                                    help='条件参数覆盖 JSON，如 \'{"rsi_level":25,"ma_period":60}\''),
    workers: int = typer.Option(8, "--workers", "-w", help="并发数"),
    no_cache: bool = typer.Option(False, "--no-cache", help="强制走网络，不用本地缓存"),
    add_tag: str = typer.Option("", "--add-to-watchlist", help="把命中结果加入自选并打上此标签"),
):
    import json as _json

    from eq.core.screener import CONDITIONS, format_screen, screen as do_screen

    conds = [c.strip() for c in conditions.split(",") if c.strip()]
    unknown = [c for c in conds if c not in CONDITIONS]
    if unknown:
        typer.echo(f"未知条件 {unknown}，可选：{', '.join(sorted(CONDITIONS))}", err=True)
        raise typer.Exit(1)
    try:
        params = _json.loads(params_json)
    except _json.JSONDecodeError as e:
        typer.echo(f"params JSON 解析失败：{e}", err=True)
        raise typer.Exit(1) from e

    # 组候选池
    try:
        symbols, pool_label = _resolve_symbols(source, top_n)
    except Exception as e:
        typer.echo(f"拉候选池失败：{e}", err=True)
        raise typer.Exit(1) from e

    if not symbols:
        typer.echo(f"候选池为空（来源：{source}）")
        raise typer.Exit(0)

    typer.echo(f"候选池：{pool_label}（{len(symbols)} 只），拉行情 + 算因子中...")
    try:
        hits = do_screen(symbols, conds, mode=mode, params=params, days=days,
                         workers=workers, use_cache=not no_cache)
    except ValueError as e:
        typer.echo(f"筛选失败：{e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(format_screen(hits, conds, mode))

    if add_tag and hits:
        added = 0
        for h in hits:
            if wl_svc.add(h["symbol"], reason=f"screen: {','.join(h['matched'])}", tags=add_tag):
                added += 1
        typer.echo(f"已把 {added} 只新命中标的加入自选（标签 {add_tag}）")


@app.command("doctor", help="环境体检：依赖/数据目录/数据库/推送通道/行情源连通性一次看全")
def doctor(
    network: bool = typer.Option(False, "--network/--no-network", help="是否真的打网络测行情源（默认不测）"),
    symbol: str = typer.Option("600519.SH", "--symbol", "-s", help="连通性测试用的标的"),
):
    from eq.core.doctor import FAIL, check, check_network, format_report

    items = check()
    if network:
        typer.echo("测行情源连通性中（可能要几秒）...")
        items += check_network(symbol)
    typer.echo(format_report(items))
    if any(i["status"] == FAIL for i in items):
        raise typer.Exit(1)


@app.command("export", help="导出自选/持仓/交易流水/规则/信号/回测记录为 CSV 或 Excel")
def export(
    datasets: str = typer.Option("", "--datasets", "-d",
                                 help="逗号分隔，缺省导全部：watchlist,positions,closed,trades,rules,signals,backtests"),
    out_dir: str = typer.Option("", "--out", "-o", help="输出目录，缺省 .eternityquant/exports/<时间戳>/"),
    fmt: str = typer.Option("csv", "--format", "-f", help="csv（一表一文件）| excel（单 xlsx 多 sheet）"),
):
    from eq.core.exporter import DATASETS, export as do_export

    names = [d.strip() for d in datasets.split(",") if d.strip()] or None
    try:
        result = do_export(names, out_dir=out_dir or None, fmt=fmt)
    except (ValueError, RuntimeError) as e:
        typer.echo(f"导出失败：{e}", err=True)
        raise typer.Exit(1) from e
    if not result["files"]:
        typer.echo("没有任何数据可导出")
        if result["skipped"]:
            typer.echo("  跳过：" + "；".join(result["skipped"]))
        return
    typer.echo(f"\n已导出到 {result['out_dir']}：")
    for name, n in result["rows"].items():
        typer.echo(f"  ✓ {DATASETS[name][1]:<10} {n:>7} 行")
    for s in result["skipped"]:
        typer.echo(f"  · 跳过 {s}")


# 子命令组：eq cache ...（行情本地缓存管理）
cache_app = typer.Typer(help="行情本地缓存管理（bar_cache）", no_args_is_help=True)
app.add_typer(cache_app, name="cache")


@cache_app.command("stats", help="看缓存了多少标的多少行、占多大")
def cache_stats(
    detail: bool = typer.Option(False, "--detail", "-d", help="逐标的列出"),
):
    from eq.data.cache import stats

    s = stats()
    typer.echo(
        f"\n行情缓存：{s['symbols']} 只标的 / {s['rows']} 行 / {s['size_mb']} MB"
        + (f"\n日期范围：{s['first_date']} ~ {s['last_date']}" if s["rows"] else "")
    )
    if detail and s["per_symbol"]:
        print(f"\n{'符号':<14} {'行数':>8}  {'起':<12} {'止':<12}")
        print("-" * 52)
        for r in s["per_symbol"]:
            print(f"{r['symbol']:<14} {r['rows']:>8}  {str(r['first']):<12} {str(r['last']):<12}")


@cache_app.command("clear", help="清空行情缓存（下次拉取会重新走网络）")
def cache_clear(
    symbol: str = typer.Option("", "--symbol", "-s", help="只清某只标的，缺省清全部"),
    yes: bool = typer.Option(False, "--yes", "-y", help="跳过确认"),
):
    from eq.data.cache import clear

    target = symbol or "全部标的"
    if not yes and not typer.confirm(f"确认清空行情缓存（{target}）？"):
        typer.echo("已取消")
        return
    n = clear(symbol or None)
    typer.echo(f"已清除 {n} 行缓存（{target}）")


@cache_app.command("warm", help="预热缓存：把自选/持仓的日线一次性拉到本地（之后离线也能跑回测/筛选）")
def cache_warm(
    source: str = typer.Option("watchlist", "--from", "-f", help="watchlist | portfolio | both | 逗号分隔代码"),
    days: int = typer.Option(400, "--days", "-d", help="每只拉多少根日线"),
    workers: int = typer.Option(8, "--workers", "-w", help="并发数"),
):
    from concurrent.futures import ThreadPoolExecutor

    from eq.data.market import get_recent_bars

    symbols, _ = _resolve_symbols(source)
    if not symbols:
        typer.echo(f"候选池为空（来源：{source}）")
        return

    typer.echo(f"预热 {len(symbols)} 只标的的 {days} 根日线...")
    ok = 0
    failed: list[str] = []

    def _one(sym: str):
        try:
            df = get_recent_bars(sym, days=days, ttl_seconds=0)  # ttl=0 强制刷新
            return sym, len(df)
        except Exception:
            return sym, 0

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(symbols)))) as pool:
        for sym, n in pool.map(_one, symbols):
            if n:
                ok += 1
                print(f"  ✓ {sym:<14} {n:>5} 行", flush=True)
            else:
                failed.append(sym)
    typer.echo(f"\n预热完成：{ok}/{len(symbols)} 成功")
    if failed:
        typer.echo(f"  失败 {len(failed)} 只：{', '.join(failed[:15])}")


# ---------- eq watchlist 子命令 ----------

@watchlist_app.command("add", help="加入自选股")
def wl_add(
    symbol: str = typer.Argument(help="股票符号，如 600519.SH"),
    reason: str = typer.Option("", "--reason", "-r", help="加入理由"),
    tags: str = typer.Option("", "--tags", "-t", help="标签，逗号分隔，如 白酒,龙头"),
):
    from eq.data.market import normalize_symbol
    try:
        norm = normalize_symbol(symbol)
    except ValueError:
        norm = symbol
    rowid = wl_svc.add(symbol, reason=reason, tags=tags)
    hint = f"（已规整：{symbol} → {norm}）" if norm != symbol else ""
    if rowid == 0:
        typer.echo(f"{norm} 已在自选列表{hint}")
    else:
        typer.echo(f"已加入自选：{norm}{hint}")


@watchlist_app.command("import", help="从文件批量导入自选股（Tab/逗号/换行分隔，自动识别 A/HK/US 代码）")
def wl_import(
    file: str = typer.Argument(help="文件路径，如 D:\\idmxz\\Table.txt"),
    reason: str = typer.Option("", "--reason", "-r", help="统一加入理由"),
    tags: str = typer.Option("", "--tags", "-t", help="统一标签，逗号分隔"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只打印将导入什么，不真正写库"),
):
    from pathlib import Path
    p = Path(str(file).strip().strip('"').strip("'"))
    if not p.exists():
        typer.echo(f"文件不存在：{file}", err=True)
        raise typer.Exit(1)
    # 品种表常见 GBK 编码（同花顺/通达信导出），按多编码依次试
    raw_bytes = p.read_bytes()
    text = None
    for enc in ("utf-8", "gbk", "gb18030", "latin-1"):
        try:
            text = raw_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        typer.echo(f"无法解码文件：{file}", err=True)
        raise typer.Exit(1)

    import re
    all_codes: list[tuple[str, str]] = []  # (code, market)
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("代码"):
            continue
        # 制表符或任意空白都当分隔符（此前硬要求含 \t，空格分隔的表全被跳过）
        raw = re.split(r"[\s\t]+", line)[0].strip()
        if not raw:
            continue
        if raw.startswith(("SH", "SZ", "BJ")):
            # qlib 格式 → 转项目符号格式
            raw_upper = raw.upper()
            if raw_upper.startswith("SH"):
                all_codes.append((f"{raw_upper[2:]}.SH", "A"))
            elif raw_upper.startswith("SZ"):
                all_codes.append((f"{raw_upper[2:]}.SZ", "A"))
            else:
                all_codes.append((f"{raw_upper[2:]}.BJ", "A"))
        elif raw.isdigit() and len(raw) == 6:
            # A 股裸 6 位
            if raw.startswith(("6", "9")):
                all_codes.append((f"{raw}.SH", "A"))
            elif raw.startswith(("0", "3")):
                all_codes.append((f"{raw}.SZ", "A"))
            elif raw.startswith(("4", "8")):
                all_codes.append((f"{raw}.BJ", "A"))
        elif raw.isdigit() and len(raw) == 5:
            # 港股裸 5 位
            all_codes.append((f"{raw}.HK", "HK"))
        elif re.match(r'^[A-Z]{1,5}$', raw):
            # 美股字母代码
            all_codes.append((f"{raw}.US", "US"))
        # 其他（指数、FX 等）跳过

    # 去重
    seen = set()
    deduped = []
    for sym, mkt in all_codes:
        if sym not in seen:
            seen.add(sym)
            deduped.append((sym, mkt))

    if not deduped:
        typer.echo(f"文件 {file} 未识别到任何股票代码")
        return

    print(f"\n识别到 {len(deduped)} 只股票（A/HK/US）：")
    for sym, mkt in deduped[:20]:
        print(f"  {sym:<12} {mkt}")
    if len(deduped) > 20:
        print(f"  ...（共 {len(deduped)} 只，仅显示前 20）")

    if dry_run:
        typer.echo("\n[dry-run] 未写库。去掉 --dry-run 真正导入。")
        return

    added = skipped = 0
    for sym, _mkt in deduped:
        rowid = wl_svc.add(sym, reason=reason, tags=tags)
        if rowid == 0:
            skipped += 1
        else:
            added += 1
    typer.echo(f"\n导入完成：新增 {added} 只，跳过 {skipped} 只（已在自选列表）")


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


@watchlist_app.command("quotes", help="一屏看完所有自选股的实时行情（并发拉取，按涨幅排序）")
def wl_quotes(
    tag: str = typer.Option("", "--tag", "-t", help="只看带某标签的自选"),
    no_cache: bool = typer.Option(False, "--no-cache", help="强制走网络，不用本地缓存"),
    realtime: bool = typer.Option(False, "--realtime", "-r", help="走实时行情源（盘中返回当前价）"),
):
    from eq.data.market import get_snapshots

    rows = wl_svc.list_by_tag(tag) if tag else wl_svc.list_all()
    if not rows:
        typer.echo(f"自选列表为空{f'（标签 {tag}）' if tag else ''}")
        return
    symbols = [r["symbol"] for r in rows]
    typer.echo(f"拉取 {len(symbols)} 只自选行情中...")
    snaps = get_snapshots(symbols, use_cache=not no_cache, realtime=realtime)
    meta = {r["symbol"]: r for r in rows}

    items = []
    for sym, snap in snaps.items():
        m = meta.get(sym, {})
        items.append({
            "symbol": sym, "name": m.get("name") or "", "tags": m.get("tags") or "",
            "snap": snap,
        })
    items.sort(key=lambda d: (d["snap"] is None, -(d["snap"]["change_pct"] if d["snap"] else 0)))

    print(f"\n自选行情（共 {len(items)} 只{f'，标签 {tag}' if tag else ''}）：\n")
    print(f"{'符号':<14} {'名称':<12} {'最新价':>10} {'涨跌幅':>10} {'成交量':>14} {'日期':<12} {'标签'}")
    print("-" * 100)
    failed = []
    for it in items:
        s = it["snap"]
        if s is None:
            failed.append(it["symbol"])
            continue
        arrow = "▲" if s["change_pct"] >= 0 else "▼"
        print(
            f"{it['symbol']:<14} {it['name'][:12]:<12} {s['close']:>10.2f} "
            f"{arrow}{s['change_pct']:>+8.2f}% {s['volume']:>14.0f} {s['date']:<12} {it['tags']}"
        )
    if failed:
        print(f"\n  ⚠ {len(failed)} 只拉取失败：{', '.join(failed[:10])}")


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
        pf_svc.add(symbol, shares, price, note=note)
        typer.echo(f"已加仓：{symbol} +{shares} 股 @ {price}")
    except ValueError as e:
        typer.echo(f"加仓失败：{e}", err=True)
        raise typer.Exit(1) from e


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
        raise typer.Exit(1) from e


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
        raise typer.Exit(1) from e


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


@portfolio_app.command("summary", help="持仓体检：一次性看盈亏/仓位占比/距止损止盈距离/今日涨跌")
def pf_summary():
    """一次性体检全持仓。

    每只持仓并发拉最新行情，算市值/浮盈/仓位占比/距止损止盈%/今日涨跌，
    末尾汇总总市值/总浮盈/总已实现/今日盈亏 + 集中度与止损覆盖率风险提示。
    """
    try:
        result = pf_svc.summary()
    except Exception as e:
        typer.echo(f"持仓体检失败：{e}", err=True)
        raise typer.Exit(1) from e
    positions = result["positions"]
    if not positions:
        typer.echo("当前无持仓，体检空")
        return
    print(f"\n持仓体检（共 {len(positions)} 只）\n")
    print(
        f"{'符号':<14} {'市场':<6} {'股数':>10} {'成本':>10} {'现价':>10} "
        f"{'市值':>12} {'浮盈':>12} {'浮盈%':>8} {'今日%':>8} {'占比%':>7} {'距止损%':>9} {'距止盈%':>9}"
    )
    print("-" * 132)
    for p in positions:
        dist_stop = f"{p['dist_to_stop_pct']:>+9.2f}" if p["dist_to_stop_pct"] is not None else f"{'-':>9}"
        dist_target = f"{p['dist_to_target_pct']:>+9.2f}" if p["dist_to_target_pct"] is not None else f"{'-':>9}"
        flag = "" if p.get("quote_ok", True) else "  ⚠行情缺失"
        print(
            f"{p['symbol']:<14} {p['market'] or '-':<6} {p['shares']:>10.0f} {p['cost_price']:>10.2f} "
            f"{p['current_price']:>10.2f} {p['market_value']:>12.2f} {p['unrealized_pnl']:>+12.2f} "
            f"{p['unrealized_pct']:>+7.2f}% {p['today_pct']:>+7.2f}% {p['weight_pct']:>6.1f}% "
            f"{dist_stop} {dist_target}{flag}"
        )
    print("-" * 132)
    print(
        f"汇总：总市值 {result['total_market_value']:,.2f}  总成本 {result['total_cost']:,.2f}  "
        f"总浮盈 {result['total_unrealized_pnl']:+,.2f}（{result['total_unrealized_pct']:+.2f}%）\n"
        f"      累计已实现 {result['total_realized_pnl']:+,.2f}  今日盈亏 {result['total_today_pnl']:+,.2f}"
    )
    # 风险提示
    warnings: list[str] = []
    if result["max_weight_pct"] > 30:
        warnings.append(
            f"集中度偏高：{result['max_weight_symbol']} 占 {result['max_weight_pct']:.1f}%（>30%）"
        )
    if result["no_stop"]:
        warnings.append(f"未设止损 {len(result['no_stop'])} 只：{', '.join(result['no_stop'][:6])}")
    if result["risk_at_stop"] > 0:
        pct = result["risk_at_stop"] / result["total_market_value"] * 100 if result["total_market_value"] else 0
        warnings.append(f"全部触发止损的合计回吐 {result['risk_at_stop']:,.0f}（占市值 {pct:.1f}%）")
    if result["stale"]:
        warnings.append(f"行情拉取失败 {len(result['stale'])} 只（用成本价占位）：{', '.join(result['stale'][:6])}")
    if warnings:
        print("\n风险提示：")
        for w in warnings:
            print(f"  · {w}")


# ---------- eq monitor 子命令 ----------

@monitor_app.command("add", help="注册监控规则")
def mon_add(
    symbol: str = typer.Argument(None, help="股票符号；省略表示全市场规则"),
    rule_type: str = typer.Argument(help=f"规则类型，可选：{','.join(sorted(mon_svc.RULE_TYPES))}"),
    params_json: str = typer.Argument("{}", help="参数 JSON，如 '{\"level\":1800,\"direction\":\"up\"}'"),
    channels: str = typer.Option("desktop", "--channels", "-c", help="推送通道，逗号分隔，如 desktop,wechat_work"),
    cooldown: int = typer.Option(0, "--cooldown", help="冷却分钟数：触发一次后 N 分钟内不重复推（0=不冷却）"),
):
    import json as _json
    try:
        params = _json.loads(params_json)
    except _json.JSONDecodeError as e:
        typer.echo(f"params JSON 解析失败：{e}", err=True)
        raise typer.Exit(1) from e
    if not isinstance(params, dict):
        typer.echo(f"params 必须是 JSON 对象，收到 {type(params).__name__}", err=True)
        raise typer.Exit(1)
    try:
        rid = mon_svc.add_rule(
            symbol or None, rule_type, params,
            channels=[c.strip() for c in channels.split(",") if c.strip()],
            cooldown_minutes=cooldown,
        )
    except ValueError as e:
        typer.echo(f"注册失败：{e}", err=True)
        raise typer.Exit(1) from e
    cd = f"  冷却={cooldown}分钟" if cooldown else ""
    typer.echo(f"已注册规则 #{rid}：{symbol or '全市场'}  {rule_type}  {params}  通道={channels}{cd}")


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
def mon_run(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="打印每条触发的规则"),
):
    fired = mon_svc.run_all(verbose=verbose)
    typer.echo(f"扫描完成，触发 {fired} 条规则")


@monitor_app.command("signals", help="查最近触发过的信号（历史回放/复盘用）")
def mon_signals(
    symbol: str = typer.Option("", "--symbol", "-s", help="按标的过滤"),
    limit: int = typer.Option(30, "--limit", "-n", help="最近 N 条"),
):
    sigs = mon_svc.recent_signals(limit=limit, symbol=symbol or None)
    if not sigs:
        typer.echo("暂无触发记录（规则触发后会自动落 signals 表）")
        return
    print(f"\n最近触发信号（{len(sigs)} 条）：\n")
    print(f"{'时间':<21} {'标的':<14} {'规则类型':<14} {'标题'}")
    print("-" * 92)
    for s in sigs:
        ctx = s.get("context") or {}
        print(f"{str(s['created_at'])[:19]:<21} {s['symbol']:<14} {s['signal_type']:<14} {ctx.get('title', '')}")


@monitor_app.command("cooldown", help="给已有规则设置/取消冷却期（分钟）")
def mon_cooldown(
    rule_id: int = typer.Argument(help="规则 id"),
    minutes: int = typer.Argument(help="冷却分钟数，0 = 取消冷却"),
):
    if mon_svc.set_cooldown(rule_id, minutes):
        typer.echo(f"规则 #{rule_id} 冷却期已设为 {minutes} 分钟" if minutes else f"规则 #{rule_id} 已取消冷却")
    else:
        typer.echo(f"规则 #{rule_id} 不存在", err=True)
        raise typer.Exit(1)


@monitor_app.command("channels", help="列出当前可用的推送通道")
def mon_channels():
    chs = available_channels()
    if not chs:
        typer.echo("无可用通道（请配置 .eternityquant/.env）")
        return
    typer.echo("可用推送通道：" + ", ".join(chs))


# ---------- eq backtest 命令 ----------

# 策略表已抽到 eq.strategy.registry（看板也要用，原来两边各抄一份）
from eq.strategy.registry import builtin_strategies as _builtin_strategies  # noqa: E402

_BUILTIN_STRATEGIES = _builtin_strategies()


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
    sweep: bool = typer.Option(False, "--sweep", help="忽略 strategy 参数，把全部内置策略都跑一遍并排名"),
    detail: bool = typer.Option(False, "--detail", help="显示完整指标（Sortino/Calmar/盈亏比/回撤天数）"),
):
    if engine not in ("vectorized", "event_driven"):
        typer.echo(f"未知引擎 {engine}，可选：vectorized / event_driven", err=True)
        raise typer.Exit(1)
    if not sweep and strategy not in _BUILTIN_STRATEGIES:
        typer.echo(f"未知策略 {strategy}，可选：{','.join(sorted(_BUILTIN_STRATEGIES))}", err=True)
        raise typer.Exit(1)
    from eq.data.market import get_recent_bars
    try:
        df = get_recent_bars(symbol, days=days)
    except Exception as e:
        typer.echo(f"拉行情失败：{e}", err=True)
        raise typer.Exit(1) from e
    if df.empty:
        typer.echo("行情为空，无法回测", err=True)
        raise typer.Exit(1)
    cfg = BacktestConfig(
        initial_cash=initial_cash,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        engine=engine,
    )

    def _new_engine():
        return VectorizedBacktester() if engine == "vectorized" else EventDrivenBacktester()

    if sweep:
        # 全策略横评：同一段行情下把内置策略都跑一遍，按夏普排名
        typer.echo(f"\n全策略横评 {symbol}（{engine}，{len(df)} 根 bar）：\n")
        rows = []
        for name in sorted(_BUILTIN_STRATEGIES):
            try:
                res = _new_engine().run(df, _BUILTIN_STRATEGIES[name], BacktestConfig(**vars(cfg)))
            except Exception as e:
                typer.echo(f"  ✗ {name} 回测失败：{e}")
                continue
            m = res.metrics
            rows.append((name, m, res))
        if not rows:
            typer.echo("全部策略均回测失败")
            raise typer.Exit(1)
        rows.sort(key=lambda t: t[1].get("sharpe", 0), reverse=True)
        print(f"{'策略':<18} {'总收益':>10} {'年化':>10} {'夏普':>8} {'Sortino':>8} "
              f"{'最大回撤':>10} {'胜率':>8} {'交易':>6}")
        print("-" * 88)
        for name, m, _ in rows:
            print(
                f"{name:<18} {m.get('total_return', 0):>+9.2%} {m.get('annual_return', 0):>+9.2%} "
                f"{m.get('sharpe', 0):>+8.2f} {m.get('sortino', 0):>+8.2f} "
                f"{m.get('max_drawdown', 0):>+9.2%} {m.get('win_rate', 0):>7.1%} {m.get('num_trades', 0):>6}"
            )
        best = rows[0]
        typer.echo(f"\n夏普最优：{best[0]}（{best[2].summary()}）")
        if save:
            from eq.backtest.store import save_result
            for name, _, res in rows:
                save_result(res, symbol=symbol, strategy_name=name)
            typer.echo(f"已把 {len(rows)} 次回测外存（`eq bt list -s {symbol}` 查看）")
        return

    result = _new_engine().run(df, _BUILTIN_STRATEGIES[strategy], cfg)
    typer.echo(f"\n回测 {symbol} 用 {strategy}（{engine}，{len(df)} 根 bar）：")
    if detail:
        typer.echo(result.detail())
    else:
        typer.echo(f"  {result.summary()}")
    if not result.trades.empty:
        typer.echo("\n交易明细（前 5 笔）：")
        print(result.trades.head(5).to_string(index=False))
    else:
        typer.echo("  （该策略在这段行情里没产生任何完整交易）")
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
        raise typer.Exit(1) from e
    meta = bundle["meta"]
    m = meta["metrics"]
    typer.echo(f"\n回测 {meta['symbol']} 用 {meta['strategy_name']}（{meta['engine']}）@ {meta['created_at']}")
    typer.echo(f"  总收益 {m.get('total_return', 0):+.2%}  年化 {m.get('annual_return', 0):+.2%}  夏普 {m.get('sharpe', 0):+.2f}  最大回撤 {m.get('max_drawdown', 0):+.2%}  胜率 {m.get('win_rate', 0):.1%}  交易 {m.get('num_trades', 0)} 笔")
    if details:
        typer.echo("\n权益曲线（前 5 日）：")
        print(bundle["equity"].head(5).to_string())
        if not bundle["trades"].empty:
            typer.echo("\n交易明细（前 5 笔）：")
            print(bundle["trades"].head(5).to_string(index=False))


@bt_app.command("remove", help="删除某次回测记录（SQLite metadata + parquet 文件）")
def bt_remove(run_id: str = typer.Argument(help="run_id")):
    from eq.backtest.store import remove_run
    if remove_run(run_id):
        typer.echo(f"已删除回测 {run_id}")
    else:
        typer.echo(f"回测 {run_id} 不存在", err=True)
        raise typer.Exit(1)


@bt_app.command("portfolio", help="组合级回测：一笔钱同时管一篮子标的（资金分配 + 持仓约束 + 真实成本）")
def bt_portfolio(
    strategy: str = typer.Argument("trend_vote", help=f"策略名，可选：{','.join(sorted(_BUILTIN_STRATEGIES))}"),
    source: str = typer.Option("watchlist", "--from", "-f", help="标的来源：watchlist|portfolio|A/HK/US|逗号分隔代码"),
    top_n: int = typer.Option(30, "--top", "-n", help="市场榜来源时取前 N 只作为候选"),
    days: int = typer.Option(500, "--days", "-d", help="回看天数"),
    cash: float = typer.Option(1_000_000, "--cash", "-c", help="初始资金"),
    max_positions: int = typer.Option(10, "--max-positions", help="最多同时持有几只"),
    max_weight: float = typer.Option(0.25, "--max-weight", help="单票权重上限"),
    allocation: str = typer.Option("equal", "--alloc", "-a", help="资金分配：equal|inverse_vol|score"),
    rebalance: str = typer.Option("signal", "--rebalance", "-r", help="调仓节奏：signal|daily|weekly|monthly"),
    costs: str = typer.Option("a_share", "--costs", help="成本模型：a_share|hk|us|crypto|flat"),
    compare: bool = typer.Option(False, "--compare", help="对比三种资金分配方式"),
    delay: int = typer.Option(1, "--exec-delay",
                              help="执行延迟(bar)。1=次日成交（散户的真实约束）；0=当日收盘成交（偏乐观）"),
    market_filter_index: str = typer.Option("", "--market-filter",
                                            help="大盘闸门指数，如 000300.SH；留空不启用"),
    ma_period: int = typer.Option(200, "--filter-ma", help="大盘闸门的均线长度"),
    workers: int = typer.Option(8, "--workers", "-w", help="拉行情并发数"),
):
    from concurrent.futures import ThreadPoolExecutor

    from eq.backtest.portfolio import (
        PortfolioConfig, compare_allocations, format_portfolio, run_portfolio,
    )
    from eq.data.market import get_recent_bars

    if strategy not in _BUILTIN_STRATEGIES:
        typer.echo(f"未知策略 {strategy}，可选：{','.join(sorted(_BUILTIN_STRATEGIES))}", err=True)
        raise typer.Exit(1)
    fn = _BUILTIN_STRATEGIES[strategy]

    try:
        symbols, label = _resolve_symbols(source, top_n)
    except Exception as e:
        typer.echo(f"拉候选池失败：{e}", err=True)
        raise typer.Exit(1) from e
    if not symbols:
        typer.echo(f"候选池为空（来源：{source}）", err=True)
        raise typer.Exit(1)

    typer.echo(f"候选池：{label}（{len(symbols)} 只），拉 {days} 根日线...")

    def _one(s):
        try:
            return s, get_recent_bars(s, days=days)
        except Exception:
            return s, None

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(symbols)))) as pool:
        bars = {s: d for s, d in pool.map(_one, symbols) if d is not None and len(d) >= 30}
    if not bars:
        typer.echo("没有拉到足够长的行情", err=True)
        raise typer.Exit(1)

    if market_filter_index:
        from eq.strategy.retail import market_filter, with_market_filter
        try:
            idx_bars = get_recent_bars(market_filter_index, days=max(days * 2, 800))
        except Exception as e:
            typer.echo(f"拉大盘指数失败：{e}", err=True)
            raise typer.Exit(1) from e
        allow = market_filter(idx_bars, ma_period=ma_period)
        typer.echo(f"大盘闸门：{market_filter_index} MA{ma_period}，"
                   f"允许持股 {allow.mean():.0%} 的时间")
        fn = with_market_filter(fn, idx_bars, ma_period=ma_period)

    cfg = PortfolioConfig(initial_cash=cash, max_positions=max_positions,
                          max_weight=max_weight, allocation=allocation,
                          rebalance=rebalance, cost_model=costs,
                          execution_delay=delay)
    try:
        if compare:
            df = compare_allocations(bars, fn, cfg)
            for c in ("总收益", "年化", "最大回撤", "年化波动"):
                df[c] = df[c].map(lambda v: f"{v:+.2%}")
            df["夏普"] = df["夏普"].map(lambda v: f"{v:+.2f}")
            df["换手x/年"] = df["换手x/年"].map(lambda v: f"{v:.1f}")
            df["平均持仓"] = df["平均持仓"].map(lambda v: f"{v:.1f}")
            print(f"\n三种资金分配方式对比（{strategy}，{len(bars)} 只候选）：\n")
            print(df.to_string(index=False))
            typer.echo("\n提示：波动率反比通常降波动、降收益；等权最简单也最不容易过拟合。")
            return
        res = run_portfolio(bars, fn, cfg)
    except ValueError as e:
        typer.echo(f"组合回测失败：{e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(format_portfolio(res))


@app.command("daily", help="每日晨报：大盘状态 + 持仓止损警报 + 今日信号 + 纸面战绩（一条命令跑完日常）")
def daily_cmd(
    strategy: str = typer.Option("trend_vote", "--strategy", "-s",
                                 help=f"信号策略，可选：{','.join(sorted(_BUILTIN_STRATEGIES))}"),
    source: str = typer.Option("both", "--from", "-f", help="信号候选池：watchlist|portfolio|both"),
    days: int = typer.Option(400, "--days", "-d", help="每只拉多少根日线"),
    horizon: int = typer.Option(10, "--horizon", help="纸面推荐的持有期（交易日）"),
    benchmark: str = typer.Option("000300.SH", "--benchmark", "-b", help="纸面战绩的比较基准"),
    record: bool = typer.Option(True, "--record/--no-record",
                                help="是否把今日买入信号记入纸面日志（默认记）"),
    workers: int = typer.Option(8, "--workers", "-w", help="拉行情并发数"),
):
    import datetime as _dt
    from concurrent.futures import ThreadPoolExecutor

    from eq.core import briefing, journal
    from eq.data.market import get_recent_bars

    if strategy not in _BUILTIN_STRATEGIES:
        typer.echo(f"未知策略 {strategy}，可选：{','.join(sorted(_BUILTIN_STRATEGIES))}", err=True)
        raise typer.Exit(1)
    fn = _BUILTIN_STRATEGIES[strategy]

    typer.echo(f"\n════ EternityQuant 晨报  {_dt.date.today().isoformat()} ════")

    # ---- 1. 大盘 ----
    idx_bars = None
    try:
        idx_bars = get_recent_bars(benchmark, days=500)
    except Exception as e:
        typer.echo(f"\n【大盘】{benchmark} 拉取失败（{e}），跳过")
    ms = briefing.market_status(idx_bars)
    if ms:
        gate = "开（允许持股）" if ms["gate_open"] else "关（大盘趋势向下）"
        line = f"\n【大盘】{benchmark}  {ms['close']:.2f}  {ms['change_pct']:+.2f}%"
        if ms["dist_ma_pct"] is not None:
            line += f"   距 MA200 {ms['dist_ma_pct']:+.1f}%"
        typer.echo(line + f"   闸门：{gate}")

    # ---- 2. 持仓警报 ----
    positions = pf_svc.list_open()
    if positions:
        try:
            summ = pf_svc.summary()
            checks = briefing.stop_breaches(summ["positions"])
            typer.echo(
                f"\n【持仓】{len(positions)} 只   市值 {summ['total_market_value']:,.0f}"
                f"   浮盈 {summ['total_unrealized_pnl']:+,.0f}"
                f"（{summ['total_unrealized_pct']:+.2f}%）   今日 {summ['total_today_pnl']:+,.0f}"
            )
            for p in checks["breached"]:
                typer.echo(f"  ‼ {p['symbol']} 已跌破止损（现价 {p['current_price']:.2f}"
                           f" ≤ 止损 {p['stop_loss']:.2f}）——按纪律该走了")
            for p in checks["near"]:
                dist = (p["current_price"] - p["stop_loss"]) / p["current_price"]
                typer.echo(f"  ⚠ {p['symbol']} 逼近止损（现价 {p['current_price']:.2f}，"
                           f"距止损 {dist:.1%}）")
            if checks["no_stop"]:
                typer.echo(f"  · 未设止损：{', '.join(p['symbol'] for p in checks['no_stop'][:8])}"
                           "（eq portfolio stops 补上）")
        except Exception as e:
            typer.echo(f"\n【持仓】体检失败：{e}")
    else:
        typer.echo("\n【持仓】空仓")

    # ---- 3. 今日信号 ----
    try:
        symbols, label = _resolve_symbols(source)
    except Exception as e:
        typer.echo(f"\n【信号】候选池解析失败：{e}")
        symbols, label = [], source
    changes: dict = {}
    bars: dict = {}
    if symbols:
        def _one(x):
            try:
                return x, get_recent_bars(x, days=days)
            except Exception:
                return x, None

        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(symbols)))) as pool:
            bars = {x: d for x, d in pool.map(_one, symbols) if d is not None and len(d) >= 30}
        changes = briefing.detect_signal_changes(bars, fn)
        enters = sorted(k for k, v in changes.items() if v == "enter")
        exits = sorted(k for k, v in changes.items() if v == "exit")
        holding = sum(1 for v in changes.values() if v == "holding")
        typer.echo(f"\n【信号】{strategy} @ {label}（{len(bars)} 只）")
        typer.echo(f"  今日新买入：{', '.join(enters) if enters else '无'}")
        typer.echo(f"  今日转卖出：{', '.join(exits) if exits else '无'}")
        typer.echo(f"  持有中 {holding} 只，其余空仓观望")
        if ms and not ms["gate_open"] and enters:
            typer.echo("  ⚠ 大盘闸门是关的——买入信号谨慎对待（回撤控制优先）")
    else:
        typer.echo("\n【信号】候选池为空（先 eq watchlist add 加几只）")

    # ---- 4. 纸面日志：记录 + 结算 + 战绩 ----
    if record and changes:
        recos, bench_px = briefing.build_recos(bars, changes, idx_bars)
        if recos:
            n = journal.record(recos, strategy, horizon_days=horizon,
                               benchmark=benchmark, benchmark_price=bench_px)
            typer.echo(f"\n【纸面】新记录 {n} 笔推荐（持有 {horizon} 交易日后自动结算）")
    try:
        settled = journal.evaluate_due()
        if settled:
            typer.echo(f"【纸面】本次结算 {len(settled)} 笔：")
            for t in settled[:8]:
                ex = f"  超额 {t['excess']:+.2%}" if t["excess"] is not None else ""
                typer.echo(f"  {t['symbol']}  {t['reco_date']}→{t['exit_date']}"
                           f"  收益 {t['ret']:+.2%}{ex}")
    except Exception as e:
        typer.echo(f"【纸面】结算失败：{e}")
    typer.echo(journal.format_scoreboard(journal.scoreboard(strategy)))


@app.command("paper", help="纸面战绩牌：策略推荐的前向真实表现 vs 基准（唯一没法作弊的验证）")
def paper_cmd(
    strategy: str = typer.Option("", "--strategy", "-s", help="按策略过滤，缺省看全部"),
    detail: int = typer.Option(0, "--detail", "-n", help="显示最近 N 笔结算明细"),
    settle: bool = typer.Option(False, "--settle", help="先把到期的推荐结算掉"),
):
    from eq.core import journal

    if settle:
        settled = journal.evaluate_due()
        typer.echo(f"结算 {len(settled)} 笔到期推荐")
    typer.echo(journal.format_scoreboard(journal.scoreboard(strategy or None)))
    if detail > 0:
        rows = journal.recent_closed(limit=detail, strategy=strategy or None)
        if rows:
            print(f"{'推荐日':<12}{'标的':<12}{'策略':<16}{'结算日':<12}{'收益':>9}{'超额':>9}")
            print("-" * 72)
            for r in rows:
                ex = f"{r['excess']:+.2%}" if r["excess"] is not None else "-"
                print(f"{str(r['reco_date'])[:10]:<12}{r['symbol']:<12}{r['strategy'][:14]:<16}"
                      f"{str(r['exit_date'])[:10]:<12}{r['ret']:>+9.2%}{ex:>9}")


@app.command("nextday", help="次日高点研究：MFE/MAE 分布 + 限价档位扫描（回答「限价该挂多高」）")
def nextday_cmd(
    source: str = typer.Option("watchlist", "--from", "-f", help="标的来源"),
    top_n: int = typer.Option(20, "--top", "-n", help="市场榜来源时取前 N 只"),
    days: int = typer.Option(600, "--days", "-d", help="回看天数"),
    targets: str = typer.Option("", "--targets", "-t",
                                help="限价档位，逗号分隔的百分数，如 0.5,1,2；缺省用一组典型值"),
    stop: float = typer.Option(0.0, "--stop", help="日内止损百分比，0=不设"),
    workers: int = typer.Option(8, "--workers", "-w", help="并发数"),
):
    from concurrent.futures import ThreadPoolExecutor

    from eq.data.market import get_recent_bars
    from eq.strategy.next_day import baseline_stats, format_baseline, simulate_limit

    try:
        symbols, label = _resolve_symbols(source, top_n)
    except Exception as e:
        typer.echo(f"拉候选池失败：{e}", err=True)
        raise typer.Exit(1) from e
    if not symbols:
        typer.echo(f"候选池为空（来源：{source}）", err=True)
        raise typer.Exit(1)

    typer.echo(f"候选池：{label}（{len(symbols)} 只），拉 {days} 根日线...")

    def _one(x):
        try:
            return x, get_recent_bars(x, days=days)
        except Exception:
            return x, None

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(symbols)))) as pool:
        bars = {x: d for x, d in pool.map(_one, symbols) if d is not None and len(d) > 60}
    if not bars:
        typer.echo("没有拉到足够长的行情", err=True)
        raise typer.Exit(1)

    stats = [baseline_stats(d) for d in bars.values()]
    stats = [s for s in stats if s.get("n")]
    keys = ("mfe_mean", "mfe_median", "mfe_p25", "mfe_p75", "mae_mean", "mae_median",
            "close_ret_mean", "close_ret_median", "overnight_mean",
            "pct_up_close", "mfe_over_mae")
    med = {k: float(np.median([s[k] for s in stats])) for k in keys}
    med["n"] = sum(s["n"] for s in stats)
    typer.echo(format_baseline(med, f"{len(bars)} 只标的的中位数"))

    default_targets = [0.003, 0.005, 0.008, 0.010, 0.015, 0.020, 0.030, 0.050]
    tg = [float(x) / 100 for x in targets.split(",") if x.strip()] if targets else []
    tg = tg or default_targets
    print("\n限价档位扫描（每天买入、次日限价止盈，A 股真实成本）：\n")
    print(f"  {'限价':<8}{'成交率':>9}{'捕获率':>9}{'净收益/笔':>12}{'胜率':>8}{'年化中位':>10}")
    print("  " + "-" * 58)
    for t in tg:
        rs = [simulate_limit(d, t, stop_pct=stop or None) for d in bars.values()]
        rs = [r.stats for r in rs if r.stats["n_trades"] > 10]
        if not rs:
            continue
        print(f"  +{t:<7.2%}{np.mean([r['fill_rate'] for r in rs]):>8.1%}"
              f"{np.mean([r['capture_rate'] for r in rs]):>9.1%}"
              f"{np.mean([r['mean_net'] for r in rs]):>+12.4%}"
              f"{np.mean([r['win_rate'] for r in rs]):>8.1%}"
              f"{np.median([r['annualized'] for r in rs]):>+10.1%}")

    from eq.backtest.cost import A_SHARE
    fee = A_SHARE.round_trip_ratio(10_000) + 2 * A_SHARE.slippage_rate
    typer.echo(
        f"\n对照：一个来回成本 {fee:.3%}，而次日收盘收益均值只有 "
        f"{med['close_ret_mean']:+.4%}。\n"
        f"要靠日内来回赚钱，你的选股必须把次日均值抬高 {fee - med['close_ret_mean']:.3%} "
        f"以上——这是个很高的门槛，先用 eq bt robust 确认你的信号有没有这个能力。"
    )


@app.command("advise", help="散户实用建议：按你的资金量算持仓数/单笔金额/换手预算")
def advise_cmd(
    capital: float = typer.Argument(help="总资金（元）"),
    market: str = typer.Option("A", "--market", "-m", help="A|HK|US|CRYPTO"),
    cost_budget: float = typer.Option(0.02, "--cost-budget",
                                      help="能接受的年化交易成本占比，默认 2%"),
):
    from eq.strategy.retail import advise, format_advice

    try:
        a = advise(capital, market, annual_cost_budget=cost_budget)
    except ValueError as e:
        typer.echo(f"{e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(format_advice(a))
    typer.echo(
        "说明：这些数字来自「最低佣金 + 印花税」的硬约束，不是经验之谈。\n"
        "  · 单笔金额低于「最低金额」时，实际费率会成倍上升\n"
        "  · 换手预算超了，就是在给券商和税务打工\n"
        "  · 想验证某个策略在你的资金量下还剩多少收益：eq bt portfolio --cash <资金>"
    )


@bt_app.command("costs", help="看各市场真实交易成本（最低佣金对小额交易的影响）")
def bt_costs(
    values: str = typer.Option("", "--values", "-v", help="成交金额，逗号分隔；缺省用一组典型值"),
):
    from eq.backtest.cost import A_SHARE, compare_costs

    vals = [float(x) for x in values.split(",") if x.strip()] or None
    df = compare_costs(vals)
    print("\n各市场实际费率（单边，单位万分之；来回% 含滑点的盈亏平衡涨幅）：\n")
    print(df.to_string(index=False))
    typer.echo(
        "\nA 股成本构成：佣金万 2.5（**每笔最低 5 元**）+ 印花税千 1（仅卖出）+ 过户费万 0.1（双边）"
    )
    typer.echo("最低佣金让小额交易的实际费率远高于名义值：")
    for v in (3_000, 10_000, 50_000):
        typer.echo(f"  成交 {v:>7,.0f} 元 → 单边 {A_SHARE.cost_ratio(v) * 1e4:>5.1f} 万分之，"
                   f"来回至少要涨 {A_SHARE.breakeven_pct(v) * 100:.3f}% 才回本")


@bt_app.command("ml", help="把训练好的 ML 模型真的跑一遍组合回测（IC 高 ≠ 能赚钱）")
def bt_ml(
    model_id: str = typer.Argument("", help="模型 id，留空用当前激活模型"),
    top_n: int = typer.Option(10, "--top", "-n", help="每期选分数最高的 N 只"),
    hold_days: int = typer.Option(5, "--hold", help="选出后至少持有几天（抑制换手）"),
    days: int = typer.Option(500, "--days", "-d", help="行情回看天数"),
    cash: float = typer.Option(1_000_000, "--cash", "-c", help="初始资金"),
    allocation: str = typer.Option("score", "--alloc", "-a", help="equal|inverse_vol|score"),
    costs: str = typer.Option("a_share", "--costs", help="成本模型"),
):
    from eq.backtest.portfolio import PortfolioConfig, format_portfolio
    from eq.strategy.ml_strategy import backtest_model

    cfg = PortfolioConfig(initial_cash=cash, max_positions=top_n,
                          allocation=allocation, cost_model=costs)
    try:
        out = backtest_model(model_id or None, top_n=top_n, hold_days=hold_days,
                             days=days, portfolio_cfg=cfg)
    except Exception as e:
        typer.echo(f"ML 组合回测失败：{e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(
        f"\n模型 {model_id or '（激活模型）'}：{out['n_pred_days']} 个预测日 / "
        f"{out['n_symbols']} 只标的 / 每期选 {out['top_n']} 只 / 持有 {out['hold_days']} 天"
    )
    typer.echo(format_portfolio(out["result"]))
    typer.echo(
        "提示：IC 衡量的是预测值与未来收益的秩相关，**不含**交易成本、换手率、"
        "持仓数约束。IC 好但这里亏钱，通常是换手太高被成本吃掉——调大 --hold 试试。"
    )


@bt_app.command("robust", help="策略稳健性验证：多标的分布 + 滚动样本外 + 随机基准（单标的单区间的结果说明不了问题）")
def bt_robust(
    strategy: str = typer.Argument(help=f"策略名，可选：{','.join(sorted(_BUILTIN_STRATEGIES))}"),
    source: str = typer.Option("watchlist", "--from", "-f",
                               help="标的来源：watchlist | portfolio | A/HK/US（市场榜）| 逗号分隔代码"),
    top_n: int = typer.Option(20, "--top", "-n", help="市场榜来源时取前 N 只"),
    days: int = typer.Option(600, "--days", "-d", help="每只标的回看天数"),
    splits: int = typer.Option(5, "--splits", help="Walk-Forward 窗口数"),
    test_bars: int = typer.Option(60, "--test-bars", help="每个窗口的测试段长度（bar）"),
    embargo: int = typer.Option(5, "--embargo", help="段间 purge 的 bar 数"),
    trials: int = typer.Option(200, "--trials", help="随机基准试验次数，0=跳过"),
    workers: int = typer.Option(8, "--workers", "-w", help="拉行情并发数"),
):
    from concurrent.futures import ThreadPoolExecutor

    from eq.backtest import robust as rb
    from eq.data.market import get_recent_bars

    if strategy not in _BUILTIN_STRATEGIES:
        typer.echo(f"未知策略 {strategy}，可选：{','.join(sorted(_BUILTIN_STRATEGIES))}", err=True)
        raise typer.Exit(1)
    fn = _BUILTIN_STRATEGIES[strategy]

    symbols, pool_label = _resolve_symbols(source, top_n)
    if not symbols:
        typer.echo(f"候选池为空（来源：{source}）", err=True)
        raise typer.Exit(1)

    typer.echo(f"候选池：{pool_label}（{len(symbols)} 只），拉 {days} 根日线...")

    def _one(sym):
        try:
            return sym, get_recent_bars(sym, days=days)
        except Exception:
            return sym, None

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(symbols)))) as pool:
        bars = {s: d for s, d in pool.map(_one, symbols) if d is not None and len(d) >= 60}
    if not bars:
        typer.echo("没有拉到足够长的行情", err=True)
        raise typer.Exit(1)

    # 1) 多标的分布
    ms = rb.multi_symbol(fn, bars)
    typer.echo(rb.format_multi_symbol(ms))

    # 2) 滚动样本外：挑数据最长的那只做代表
    longest = max(bars, key=lambda s: len(bars[s]))
    wf = rb.walk_forward(bars[longest], fn, n_splits=splits,
                         test_bars=test_bars, embargo_bars=embargo)
    typer.echo(f"（Walk-Forward 用数据最长的 {longest}）")
    typer.echo(rb.format_walk_forward(wf))

    # 3) 随机基准
    if trials > 0:
        r = rb.random_benchmark(bars[longest], fn, n_trials=trials)
        typer.echo(
            f"随机基准（{longest}，{r['n_trials']} 次同频随机进出场）：\n"
            f"  真实夏普 {r['actual']:+.2f}   随机均值 {r['random_mean']:+.2f}"
            f" ± {r['random_std']:.2f}   百分位 {r['percentile']:.0f}   p={r['p_value']:.3f}\n"
            f"  判定：{r['verdict']}\n"
        )


@bt_app.command("optimize", help="参数寻优：样本内选参 + 样本外验证 + 参数高原检查（防过拟合）")
def bt_optimize(
    symbol: str = typer.Argument(help="标的，如 600519.SH"),
    strategy: str = typer.Argument("ema_cross", help="策略名（需支持参数的那几个）"),
    grid: str = typer.Option("", "--grid", "-g",
                             help='参数网格 JSON，如 \'{"fast":[3,5,8],"slow":[20,30,60]}\'；'
                                  "留空用该策略的默认网格"),
    days: int = typer.Option(800, "--days", "-d", help="回看天数"),
    test_ratio: float = typer.Option(0.3, "--test-ratio", help="样本外占比"),
    embargo: int = typer.Option(5, "--embargo", help="样本内外之间 purge 的 bar 数"),
    metric: str = typer.Option("sharpe", "--metric", "-m", help="优化目标：sharpe|calmar|total_return"),
    show: int = typer.Option(8, "--show", help="展示前 N 组参数"),
):
    import json as _json
    from functools import partial

    from eq.backtest import robust as rb
    from eq.data.market import get_recent_bars
    from eq.strategy import signals as S

    # 可寻优的策略 → (函数, 默认网格)
    tunable = {
        "ema_cross": (S.ema_cross, {"fast": [3, 5, 8, 12, 20], "slow": [20, 30, 40, 60, 90]}),
        "adx_trend": (S.adx_trend, {"period": [10, 14, 20], "threshold": [20.0, 25.0, 30.0]}),
        "rsi_reversal": (S.rsi_reversal, {"period": [7, 14, 21],
                                          "oversold": [20.0, 25.0, 30.0],
                                          "overbought": [70.0, 75.0, 80.0]}),
        "bollinger_break": (S.bollinger_break, {"period": [10, 20, 30], "k": [1.5, 2.0, 2.5]}),
        "donchian": (S.donchian_breakout, {"entry": [10, 20, 55], "exit_period": [5, 10, 20]}),
        "keltner": (S.keltner_breakout, {"period": [10, 20, 30], "mult": [1.5, 2.0, 3.0]}),
        "supertrend": (S.supertrend_follow, {"period": [7, 10, 14], "mult": [2.0, 3.0, 4.0]}),
        "zscore_reversion": (S.zscore_reversion, {"period": [10, 20, 40],
                                                  "entry": [1.5, 2.0, 2.5],
                                                  "exit_z": [0.0, 0.5, 1.0]}),
        "vol_breakout": (S.volatility_breakout, {"period": [10, 20, 30], "k": [0.3, 0.5, 0.8]}),
    }
    if strategy not in tunable:
        typer.echo(f"策略 {strategy} 不支持参数寻优，可选：{','.join(sorted(tunable))}", err=True)
        raise typer.Exit(1)
    base_fn, default_grid = tunable[strategy]
    if grid:
        try:
            g = _json.loads(grid)
        except _json.JSONDecodeError as e:
            typer.echo(f"网格 JSON 解析失败：{e}", err=True)
            raise typer.Exit(1) from e
    else:
        g = default_grid

    try:
        df = get_recent_bars(symbol, days=days)
    except Exception as e:
        typer.echo(f"拉行情失败：{e}", err=True)
        raise typer.Exit(1) from e

    n_combos = 1
    for v in g.values():
        n_combos *= len(v)
    typer.echo(f"\n{symbol} {len(df)} 根 bar，{strategy} 参数网格 {n_combos} 组，"
               f"样本内 {1 - test_ratio:.0%} 选参 / 样本外 {test_ratio:.0%} 验证\n")

    try:
        res = rb.optimize(df, lambda **kw: partial(base_fn, **kw), g,
                          test_ratio=test_ratio, embargo_bars=embargo, metric=metric)
    except ValueError as e:
        typer.echo(f"寻优失败：{e}", err=True)
        raise typer.Exit(1) from e

    sweep = res["sweep"].drop(columns=["_combo"], errors="ignore")
    cols = [c for c in sweep.columns
            if c in g or c in ("total_return", "sharpe", "calmar", "max_drawdown", "num_trades")]
    print(f"样本内前 {min(show, len(sweep))} 组：\n")
    print(sweep[cols].head(show).to_string(index=False))

    typer.echo(
        f"\n最优参数：{res['best_params']}\n"
        f"  样本内 {metric} {res['in_sample_metric']:+.3f}"
        f"  →  样本外 {res['out_of_sample_metric']:+.3f}"
        f"   衰减 {res['degradation']:.0%}   参数高原分 {res['plateau']:.2f}\n"
        f"  判定：{res['verdict']}"
    )
    typer.echo(
        "\n提示：参数高原分低于 0.3 说明最优点是孤立尖峰（旁边一格就掉下去），"
        "这种参数换个市场/时段基本必然失效。"
    )


@bt_app.command("compare", help="并排比较多次回测（按 run_id，或用 --symbol 比某标的的全部回测）")
def bt_compare(
    run_ids: str = typer.Argument("", help="逗号分隔的 run_id；留空则配合 --symbol 用"),
    symbol: str = typer.Option("", "--symbol", "-s", help="比较该标的最近 N 次回测"),
    limit: int = typer.Option(10, "--limit", "-n", help="--symbol 模式下取最近 N 条"),
    sort_by: str = typer.Option("sharpe", "--by", "-b", help="排序键：sharpe|total_return|calmar|max_drawdown"),
):
    from eq.backtest.store import list_runs, load_result

    metas = []
    ids = [r.strip() for r in run_ids.split(",") if r.strip()]
    if ids:
        for rid in ids:
            try:
                metas.append(load_result(rid)["meta"])
            except KeyError:
                typer.echo(f"  ⚠ 跳过不存在的 run_id：{rid}", err=True)
    elif symbol:
        metas = list_runs(symbol=symbol, limit=limit)
    else:
        typer.echo("请给出 run_id 列表，或用 --symbol 指定标的", err=True)
        raise typer.Exit(1)

    if not metas:
        typer.echo("没有可比较的回测记录")
        raise typer.Exit(1)

    reverse = sort_by != "max_drawdown"  # 回撤是负数，越大（越接近 0）越好
    metas.sort(key=lambda m: (m.get("metrics") or {}).get(sort_by, 0), reverse=reverse)

    print(f"\n回测横评（按 {sort_by} 排序，{len(metas)} 条）：\n")
    print(f"{'run_id':<24} {'标的':<12} {'策略':<16} {'引擎':<13} {'总收益':>10} "
          f"{'年化':>10} {'夏普':>8} {'回撤':>9} {'胜率':>7} {'交易':>5}")
    print("-" * 122)
    for m in metas:
        k = m.get("metrics") or {}
        print(
            f"{m['id']:<24} {m['symbol']:<12} {m['strategy_name']:<16} {m['engine']:<13} "
            f"{k.get('total_return', 0):>+9.2%} {k.get('annual_return', 0):>+9.2%} "
            f"{k.get('sharpe', 0):>+8.2f} {k.get('max_drawdown', 0):>+8.2%} "
            f"{k.get('win_rate', 0):>6.1%} {k.get('num_trades', 0):>5}"
        )
    best = metas[0]
    typer.echo(f"\n最优：{best['strategy_name']} @ {best['symbol']}（run_id={best['id']}）")


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
        raise typer.Exit(1) from e
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


@ml_app.command("train", help="走 qlib workflow 真训练（Alpha158 + LightGBM/PyTorch/DeepLOB/TFT，可选 GPU/CUDA）")
def ml_train(
    universe: str = typer.Argument("csi300", help="标的池，如 csi300/csi500/all/watchlist"),
    horizon: int = typer.Argument(5, help="预测窗口（天）"),
    algo: str = typer.Option("lightgbm", "--algo", "-a", help="lightgbm | alstm | gru | lstm | mlp | deeplob | tft"),
    train_start: str = typer.Option("2015-01-01", "--train-start", help="训练区间起"),
    train_end: str = typer.Option("2020-08-31", "--train-end", help="训练区间止"),
    valid_start: str = typer.Option("2020-09-01", "--valid-start", help="验证区间起"),
    valid_end: str = typer.Option("2020-09-25", "--valid-end", help="验证区间止（qlib 数据末日）"),
    auto_range: bool = typer.Option(True, "--auto-range/--no-auto-range", help="自动检测 qlib 数据实际可用区间覆盖默认值（默认开）"),
    device: str = typer.Option("cpu", "--device", "-d", help="cpu | gpu | cuda（LightGBM gpu=OpenCL；PyTorch cuda=真CUDA，CUDA GPU主场）"),
    hidden: int = typer.Option(0, "--hidden", help="RNN/Transformer 隐藏层大小，0=自动（GRU=64, DeepLOB=64, TFT=256）"),
    layers: int = typer.Option(0, "--layers", help="RNN 层数，0=自动（默认 2）"),
    batch: int = typer.Option(0, "--batch", "-b", help="batch size，0=自动（默认 4000，DeepLOB 建议 512，TFT 建议 256）"),
    name: str = typer.Option("", "--name", "-n", help="模型名，默认自动生成"),
    # --- 高级参数 ---
    optimizer: str = typer.Option("lion", "--optimizer", "-o", help="优化器: lion（默认，省显存+抗噪）| adamw | sam | lookahead"),
    loss: str = typer.Option("sharpe", "--loss", "-l", help="损失函数: sharpe | mse | ic"),
    dropout: float = typer.Option(0.3, "--dropout", help="Dropout 率（量化建议 0.3-0.4）"),
    adversarial: bool = typer.Option(False, "--adversarial/--no-adv", help="FGSM 对抗训练（增强鲁棒性，训练时间翻倍）"),
    orthogonalize: bool = typer.Option(False, "--orthogonalize/--no-orth", help="特征正交化去 Beta"),
    seq_len: int = typer.Option(0, "--seq-len", help="DeepLOB/TFT 输入窗口，0=自动（DeepLOB=120, TFT=60）"),
    heads: int = typer.Option(4, "--heads", help="TFT 注意力头数"),
    gpus: str = typer.Option("", "--gpus", help="多卡并行GPU ID，如 '0,1,2,3'（默认单卡）"),
    # --- v0.25 训练策略参数 ---
    features: str = typer.Option("Alpha158", "--features", "-F",
                                 help="特征集：Alpha158（截面因子，适合 lightgbm/mlp）| Alpha360（6 价量字段×60天，真时序，RNN 官方配置）"),
    test_ratio: float = typer.Option(0.2, "--test-ratio",
                                     help="从验证区间尾部切出的独立测试段占比。0=不切（成绩会偏高）"),
    embargo: int = typer.Option(-1, "--embargo",
                                help="段间 purge 的交易日数，-1=自动取 horizon（标签用到 T+h 的价格，不 purge 就是泄漏）"),
    seed: int = typer.Option(42, "--seed", help="随机种子（此前无种子控制，两次跑结果不可比）"),
    lr: float = typer.Option(-1.0, "--lr",
                             help="学习率，-1=按优化器取默认（lion 1e-4 / adamw 1e-3）。"
                                  "Lion 是符号更新，步长和梯度大小无关，需比 AdamW 小一个量级"),
    weight_decay: float = typer.Option(-1.0, "--weight-decay",
                                       help="权重衰减，-1=按优化器取默认（lion 1e-4 / adamw 1e-5）"),
    seeds: int = typer.Option(1, "--seeds",
                              help="多种子集成：跑 N 次同配置不同种子取平均。"
                                   "低信噪比数据上单次训练方差极大（同超参换个种子 IC 能差一倍），"
                                   "集成降方差不靠调参运气。建议 3~5，训练时间线性增加"),
):
    # torch DLL 预热（Windows + cu132 坑：qlib 集成链触发 torch 延迟加载 c10.dll 失败，ml 命令才预热）
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.init()
    except ImportError:
        pass
    from eq.strategy.factors.ml_workflow import train as wf_train, train_torch as wf_train_torch, _TORCH_ALGOS
    _ADVANCED_ALGOS = {"deeplob", "tft"}

    # 自动检测 qlib 数据实际可用区间，覆盖与数据不重叠的默认区间（2015~2020）。
    # 只在 auto_range=True 且用户未显式改默认值时触发；
    # 检测失败（如日历缺失）退化为用默认值，不阻断训练。
    if auto_range:
        try:
            from eq.data.paths import QLIB_CN_DATA_DIR
            from pathlib import Path as _P
            cal = _P(QLIB_CN_DATA_DIR) / "calendars" / "day.txt"
            if cal.exists():
                days = [ln.strip() for ln in cal.read_text().splitlines() if ln.strip()]
                if days:
                    days.sort()
                    data_start, data_end = days[0], days[-1]
                    # 仅当默认区间与数据区间不重叠时才覆盖，避免抹掉用户显式传的值
                    default_conflict = (train_end < data_start) or (valid_end < data_start)
                    if default_conflict:
                        # 数据末 60 个交易日作验证（不少于数据 10%），其余作训练。
                        # 之前 30 日太短导致 LightGBM 18 步就 early stop（验证集样本不足）。
                        valid_n = max(60, len(days) // 10)
                        valid_n = min(valid_n, len(days) // 3)  # 不超过数据 1/3
                        valid_start = days[-valid_n]
                        valid_end = data_end
                        train_end = days[-(valid_n + 1)]
                        train_start = data_start
                        typer.echo(
                            f"  [auto-range] 检测到数据区间 {data_start}~{data_end}（{len(days)} 日），"
                            f"自动切分 train={train_start}~{train_end}（{len(days)-valid_n} 日）"
                            f" valid={valid_start}~{valid_end}（{valid_n} 日）"
                        )
        except Exception as _e:
            typer.echo(f"  [warn] auto-range 检测失败：{_e}，用默认区间")

    try:
        if algo in _ADVANCED_ALGOS:
            # 高级模型（DeepLOB / TFT）：用 AdvancedTrainer
            kw = {}
            if hidden > 0:
                kw["hidden_size"] = hidden
            if batch > 0:
                kw["batch_size"] = batch
            if seq_len > 0:
                kw["seq_len"] = seq_len
            if heads > 0:
                kw["num_heads"] = heads
            result = wf_train_torch(
                universe=universe, horizon=horizon, algo=algo,
                train_start=train_start, train_end=train_end,
                valid_start=valid_start, valid_end=valid_end,
                device=device, name=name or None,
                optimizer=optimizer, loss_type=loss,
                dropout=dropout, adversarial=adversarial,
                orthogonalize=orthogonalize,
                gpu_ids=gpus if gpus else None,
                test_ratio=test_ratio, embargo_days=(None if embargo < 0 else embargo),
                seed=seed, feature_set=features,
                lr=(None if lr < 0 else lr),
                weight_decay=(None if weight_decay < 0 else weight_decay), **kw,
            )
        elif algo in _TORCH_ALGOS:
            # PyTorch 模型默认 cuda（GPU 参数透传给 qlib，cuda → GPU=0）
            kw = {}
            if hidden > 0:
                kw["hidden_size"] = hidden
            if layers > 0:
                kw["num_layers"] = layers
            if batch > 0:
                kw["batch_size"] = batch
            result = wf_train_torch(
                universe=universe, horizon=horizon, algo=algo,
                train_start=train_start, train_end=train_end,
                valid_start=valid_start, valid_end=valid_end,
                device=device, name=name or None, dropout=dropout,
                # optimizer 此前**没传**——gru/lstm/mlp 走这个分支，
                # `--optimizer adamw` 静默失效，永远用默认 lion
                optimizer=optimizer,
                gpu_ids=gpus if gpus else None,
                test_ratio=test_ratio, embargo_days=(None if embargo < 0 else embargo),
                seed=seed, feature_set=features,
                lr=(None if lr < 0 else lr),
                weight_decay=(None if weight_decay < 0 else weight_decay),
                n_seeds=seeds, **kw,
            )
        else:
            result = wf_train(
                universe=universe, horizon=horizon, algo=algo,
                train_start=train_start, train_end=train_end,
                valid_start=valid_start, valid_end=valid_end,
                device=device, name=name or None,
                test_ratio=test_ratio, embargo_days=(None if embargo < 0 else embargo),
                seed=seed, feature_set=features,
            )
    except Exception as e:
        typer.echo(f"训练失败：{e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"\n训练完成：model_id={result['model_id']}")
    _m = result["metrics"]
    _test = _m.get("test")
    if _test:
        from eq.strategy.factors.evaluation import verdict
        typer.echo(
            f"  测试段 Rank IC={_m['ic']:+.4f}  ICIR={_test.get('icir', 0):+.2f}  "
            f"t={_test.get('t_stat', 0):+.2f}  （验证段最优 {_m.get('valid_ic', 0):+.4f}，仅供参考）"
        )
        typer.echo(f"  判定：{verdict(_test)}")
    else:
        typer.echo(
            f"  IC={_m['ic']:+.4f}  ⚠ 无独立测试段，这是**选择集**上的最优值、偏乐观；"
            f"加长验证区间或调大 --test-ratio"
        )
    typer.echo(f"  algo={algo}  device={device}  seed={seed}  特征集={features}  模型={result['model_path']}")
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
        raise typer.Exit(1) from e
    if df.empty:
        typer.echo("预测结果为空")
        return
    # trend 列由 _score_to_trend 加上（强多/弱多/中性/弱空/强空 + trend_prob 信心分）
    has_trend = "trend" in df.columns
    print(f"\n预测前 {len(df)} 名（已写入 ml_predictions 表）：\n")
    if has_trend:
        print(f"{'符号':<14} {'分数':>9} {'趋势':<6} {'信心':>6}")
        print("-" * 40)
        for _, r in df.iterrows():
            print(f"{r['symbol']:<14} {r['score']:>+9.4f} {r['trend']:<6} {float(r['trend_prob']):>6.2f}")
    else:
        print(df.to_string(index=False))


@ml_app.command("top", help="查某日模型预测榜（按分数降序，回看命中用）")
def ml_top(
    date: str = typer.Argument(help="YYYY-MM-DD，如 2026-07-21"),
    model_id: str = typer.Option("", "--model", "-m", help="按模型过滤，缺省取所有"),
    top_n: int = typer.Option(20, "--top", "-n", help="前 N 名"),
):
    from eq.db import execute
    if model_id:
        rows = execute(
            "SELECT symbol, score, model_id FROM ml_predictions "
            "WHERE date = ? AND model_id = ? ORDER BY score DESC LIMIT ?",
            (date, model_id, top_n),
        )
    else:
        rows = execute(
            "SELECT symbol, score, model_id FROM ml_predictions "
            "WHERE date = ? ORDER BY score DESC LIMIT ?",
            (date, top_n),
        )
    if not rows:
        typer.echo(f"{date} 无预测记录（model_id={model_id or '全部'}）")
        return
    print(f"\n{date} 预测榜（model_id={model_id or '全部'}，前 {len(rows)} 名）：\n")
    print(f"{'排名':<4} {'符号':<14} {'分数':>10} {'模型':<14}")
    print("-" * 50)
    for i, r in enumerate(rows, 1):
        print(f"{i:<4} {r['symbol']:<14} {r['score']:>+10.4f} {r['model_id']:<14}")


@ml_app.command("update-data", help="更新 qlib 本地数据到最新（腾讯 API 拉 A 股日线 → 续 .bin，多线程并行）")
def ml_update_data(
    start: str = typer.Option("2020-09-28", "--start", "-s", help="续期起始日，默认接 2020-09-25"),
    end: str = typer.Option("", "--end", "-e", help="续期结束日，默认今天"),
    universe: str = typer.Option("csi300", "--universe", "-u", help="csi300/csi500/all/watchlist（watchlist 从 D:\\idmxz\\Table.txt 读取）"),
    extra: str = typer.Option("", "--extra", "-x", help="额外股票代码，逗号分隔，如 SH600519,SZ000001（与 universe 合并下载训练）"),
    workers: int = typer.Option(8, "--workers", "-w", help="并行进程数（腾讯 API 国内直连限流宽松，默认 8，上限建议 16）"),
):
    from eq.strategy.factors.ml_data_updater import update_qlib_data
    extra_codes = [c.strip().upper() for c in extra.split(",") if c.strip()] if extra else None
    try:
        result = update_qlib_data(
            start=start, end=end or None, universe=universe,
            extra_codes=extra_codes, workers=workers, verbose=True,
        )
    except Exception as e:
        typer.echo(f"更新失败：{e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"\n更新完成：续 {result['trading_days']} 交易日，{result['instruments_updated']} 只票 × {result['features_per_inst']} 特征")
    from eq.data.paths import QLIB_CN_DATA_DIR
    typer.echo(f"  数据目录：{QLIB_CN_DATA_DIR}")
    typer.echo(f"  日历新增 {result['days_added']} 行，现在可以 `eq ml train` 用最新数据训练了")


@ml_app.command("regen-instruments", help="不重下数据，仅重建 instruments/<universe>.txt（修训练 universe 无数据用）")
def ml_regen_instruments(
    universe: str = typer.Argument("csi300", help="csi300/csi500/all/watchlist"),
    extra: str = typer.Option("", "--extra", "-x", help="额外股票代码，逗号分隔，与 universe 合并"),
):
    from eq.strategy.factors.ml_data_updater import _generate_instruments, _tencent_instruments
    instruments = _tencent_instruments(universe)
    extra_codes = [c.strip().upper() for c in extra.split(",") if c.strip()] if extra else []
    merged = list(instruments)
    for c in extra_codes:
        if c not in merged:
            merged.append(c)
    try:
        _generate_instruments(universe, merged, verbose=True)
    except Exception as e:
        typer.echo(f"重建失败：{e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"\n重建完成：instruments/{universe}.txt ({len(merged)} 只)")
    typer.echo(f"  现在可以 `eq ml train {universe} <horizon> --algo <algo>` 了")


@ml_app.command("search", help="LSTM 超参网格搜索（自动试 hidden/layers/lr/batch 组合，报告 Top3）")
def ml_search(
    universe: str = typer.Argument("csi300", help="标的池 csi300/csi500/all/watchlist"),
    horizon: int = typer.Argument(5, help="预测窗口（天）"),
    algo: str = typer.Option("gru", "--algo", "-a", help="gru | lstm"),
    fast: bool = typer.Option(True, "--fast/--full", help="快速模式 max_steps=50 还是完整模式 200"),
    auto: bool = typer.Option(False, "--auto", help="搜索后自动用最佳参数全量训练（注意 -a 可能是 --algo）"),
    device: str = typer.Option("cuda", "--device", "-d", help="cuda/cpu"),
):
    from eq.strategy.factors.ml_workflow import search_lstm
    try:
        results = search_lstm(universe=universe, horizon=horizon, fast=fast, device=device, auto_train=auto, algo=algo)
        if not results:
            typer.echo("无有效结果")
            raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"搜索失败：{e}", err=True)
        raise typer.Exit(1) from e


# 注：这里原本有一段 `if __name__ == "__main__": app()`。
# 它位于文件中段，`python -m eq.cli` 时会在 scheduler/hk/data/dash 这些
# 子命令注册之前就把 app 跑起来，导致这些命令在直接执行模块时全部"不存在"。
# 已挪到文件末尾。


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
        raise typer.Exit(1) from e
    try:
        jid = sched_svc.add_job(
            name=name, cron_expr=cron_expr, action=action,
            params=params, channels=[c.strip() for c in channels.split(",") if c.strip()],
        )
    except ValueError as e:
        typer.echo(f"注册失败：{e}", err=True)
        raise typer.Exit(1) from e
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


# ---------- eq hk 子命令 ----------

@hk_app.command("update-data", help="下载港股日线数据到本地缓存（Sina 源，200 只约 3 分钟）")
def hk_update_data(
    start: str = typer.Option("", "--start", "-s", help="起始日，默认 2 年前"),
    end: str = typer.Option("", "--end", "-e", help="结束日，默认今天"),
    top_n: int = typer.Option(200, "--top", "-n", help="前 N 只热门股（显式 --codes-file 时仅截前 N）"),
    workers: int = typer.Option(3, "--workers", "-w", help="并行数"),
    codes_file: str = typer.Option("", "--codes-file", help="自定义品种表 txt 路径，自动解析其中港股代码（5 位数字）"),
):
    from eq.data.hk_market import update_hk_data, parse_hk_codes_from_file
    try:
        codes = None
        if codes_file:
            codes = parse_hk_codes_from_file(codes_file)
            if not codes:
                typer.echo(f"警告：品种表 {codes_file} 未解析出港股代码，回退到热门榜", err=True)
                codes = None
        result = update_hk_data(
            start=start or None, end=end or None, top_n=top_n,
            workers=workers, codes=codes,
        )
    except Exception as e:
        typer.echo(f"港股数据下载失败：{e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"\n港股数据更新完成：{result['codes']} 只 ✓，缓存目录：{result.get('cache_dir','?')}")


@hk_app.command("train", help="港股 GRU 训练（自写特征 ~60 维 + RecurrentAlphaNet）")
def hk_train(
    top_n: int = typer.Option(100, "--top", "-n", help="前 N 只热门股"),
    horizon: int = typer.Option(5, "--horizon", "-h", help="预测窗口（天）"),
    cell_type: str = typer.Option("gru", "--cell", "-c", help="gru/lstm"),
    hidden_size: int = typer.Option(128, "--hidden", help="隐藏层大小"),
    num_layers: int = typer.Option(2, "--layers", help="层数"),
    dropout: float = typer.Option(0.3, "--dropout", help="Dropout 率（量化建议 0.3-0.4）"),
    walk_forward: bool = typer.Option(True, "--walk-forward/--no-walk", help="Walk-Forward 滚动验证"),
    device: str = typer.Option("cuda", "--device", "-d", help="cuda/cpu"),
    optimizer: str = typer.Option("lion", "--optimizer", "-o", help="优化器: lion（默认，省显存+抗噪）| adamw"),
    name: str = typer.Option("", "--name", help="模型名"),
    gpus: str = typer.Option("", "--gpus", help="多卡并行GPU ID，如 '0,1,2,3'（默认单卡）"),
    test_ratio: float = typer.Option(0.2, "--test-ratio", help="独立测试段占比，0=不切（成绩偏高）"),
    seed: int = typer.Option(42, "--seed", help="随机种子"),
    cs_norm: bool = typer.Option(True, "--cs-norm/--raw-label",
                                 help="标签按日横截面 rank 归一化（默认开）。关掉的话模型学的是预测大盘而非选股"),
):
    from eq.data.hk_market import train_hk
    try:
        result = train_hk(
            top_n=top_n, horizon=horizon,
            cell_type=cell_type, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout,
            walk_forward=walk_forward, device=device,
            optimizer=optimizer,
            name=name or None,
            gpu_ids=gpus if gpus else None,
            test_ratio=test_ratio, seed=seed, cs_normalize_label=cs_norm,
        )
    except Exception as e:
        typer.echo(f"港股训练失败：{e}", err=True)
        raise typer.Exit(1) from e
    rep = result.get("test_report")
    if rep:
        from eq.strategy.factors.evaluation import verdict
        typer.echo(
            f"\n港股训练完成：测试段 Rank IC={result['ic']:+.4f}  ICIR={rep.get('icir', 0):+.2f}"
            f"  （验证段 {result['valid_ic']:+.4f}，仅供参考）"
        )
        typer.echo(f"  判定：{verdict(rep)}")
    else:
        typer.echo(f"\n港股训练完成：IC={result['ic']:+.4f}  ⚠ 无独立测试段，偏乐观")
    if result.get("wf_windows"):
        typer.echo(
            f"  Walk-Forward（{result['wf_windows']} 窗口）：Rank IC 均值 {result['wf_ic_mean']:+.4f}"
            f"  标准差 {result['wf_ic_std']:.4f}  区间 [{result['wf_ic_min']:+.4f}, {result['wf_ic_max']:+.4f}]"
        )
    typer.echo(f"  {result['symbols']} 只 / {result['trading_days']} 交易日  模型：{result['model_path']}")


@hk_app.command("predict", help="用港股模型批量预测 TopN")
def hk_predict(
    model_path: str = typer.Argument(help="模型文件路径"),
    top_n: int = typer.Option(10, "--top", "-n", help="前 N 名"),
):
    from eq.data.hk_market import predict_hk_top
    try:
        df = predict_hk_top(model_path=model_path, top_n=top_n)
    except Exception as e:
        typer.echo(f"港股预测失败：{e}", err=True)
        raise typer.Exit(1) from e
    if df.empty:
        typer.echo("预测结果为空")
        return
    print(f"\n港股预测 Top{top_n}：\n")
    print(df.to_string(index=False))


@hk_app.command("train-minute", help="港股分钟线 GRU 训练（走本地缓存 5m/1m，方案 A）")
def hk_train_minute(
    freq: str = typer.Option("5m", "--freq", "-f", help="5m | 1m"),
    top_n: int = typer.Option(100, "--top", "-n", help="前 N 只"),
    horizon: int = typer.Option(30, "--horizon", "-h", help="预测窗口（根数，5m×30=2.5h）"),
    cell_type: str = typer.Option("gru", "--cell", "-c", help="gru/lstm"),
    hidden_size: int = typer.Option(128, "--hidden", help="隐藏层大小"),
    num_layers: int = typer.Option(2, "--layers", help="层数"),
    dropout: float = typer.Option(0.3, "--dropout", help="Dropout 率"),
    walk_forward: bool = typer.Option(True, "--walk-forward/--no-walk", help="Walk-Forward 滚动验证"),
    device: str = typer.Option("cuda", "--device", "-d", help="cuda/cpu"),
    name: str = typer.Option("", "--name", help="模型名"),
    gpus: str = typer.Option("", "--gpus", help="多卡并行 GPU ID，如 '0,1'"),
):
    from eq.data.hk_market import train_hk_minute
    try:
        result = train_hk_minute(
            freq=freq, top_n=top_n, horizon=horizon,
            cell_type=cell_type, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout,
            walk_forward=walk_forward, device=device,
            name=name or None,
            gpu_ids=gpus if gpus else None,
        )
    except Exception as e:
        typer.echo(f"分钟线训练失败：{e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(
        f"\n分钟线训练完成({result['freq']})：IC={result['ic']:+.4f}  "
        f"{result['symbols']} 只  模型：{result['model_path']}"
    )


@hk_app.command("predict-ensemble", help="港股多频率集成预测（方案 A：日线 + 5m + 1m 加权）")
def hk_predict_ensemble(
    model_daily: str = typer.Argument(help="日线模型路径（必传）"),
    model_5m: str = typer.Option("", "--5m", help="5 分钟模型路径"),
    model_1m: str = typer.Option("", "--1m", help="1 分钟模型路径"),
    top_n: int = typer.Option(10, "--top", "-n", help="前 N 名"),
    w_daily: float = typer.Option(1.0, "--w-daily", help="日线权重"),
    w_5m: float = typer.Option(1.0, "--w-5m", help="5 分钟权重"),
    w_1m: float = typer.Option(1.0, "--w-1m", help="1 分钟权重"),
    lookback_days: int = typer.Option(90, "--lookback", help="日线在线补拉的回看天数"),
):
    from eq.data.hk_market import predict_hk_ensemble
    weights = {"daily": w_daily, "5m": w_5m, "1m": w_1m}
    try:
        df = predict_hk_ensemble(
            model_daily=model_daily,
            model_5m=model_5m or None,
            model_1m=model_1m or None,
            top_n=top_n,
            weights=weights,
            lookback_days=lookback_days,
        )
    except Exception as e:
        typer.echo(f"集成预测失败：{e}", err=True)
        raise typer.Exit(1) from e
    if df.empty:
        typer.echo("集成预测结果为空")
        return
    print(f"\n港股多频率集成预测 Top{top_n}：\n")
    print(df.to_string(index=False))


# ---------- eq dash 命令（放末尾，避免 streamlit 启动逻辑被其他装饰器干扰） ----------

# ===== eq data 数据收集命令 =====

@data_app.command("a", help="A 股日线（腾讯 API → qlib .bin）")
def data_a(
    start: str = typer.Option("2026-01-01", "--start", "-s", help="起始日"),
    universe: str = typer.Option("csi300", "--universe", "-u", help="csi300/csi500/all/watchlist（watchlist 从 D:\\idmxz\\Table.txt 读取）"),
    extra: str = typer.Option("", "--extra", "-x", help="额外股票代码，逗号分隔，如 SH600519,SZ000001（与 universe 合并下载训练）"),
    workers: int = typer.Option(5, "--workers", "-w", help="并行进程数（默认5，过大易被API限流）"),
):
    from eq.data.collector import collect_a_share
    extra_codes = [c.strip().upper() for c in extra.split(",") if c.strip()] if extra else None
    collect_a_share(start=start, universe=universe, workers=workers, extra_codes=extra_codes)


@data_app.command("hk", help="港股日线（东财 push2his 主源，akshare fallback）")
def data_hk(
    top_n: int = typer.Option(200, "--top", "-n", help="前 N 只"),
    start: str = typer.Option("2024-01-01", "--start", "-s", help="起始日"),
    codes: str = typer.Option("", "--codes", help="显式指定港股代码（逗号分隔，如 00700,09988），优先于 top"),
):
    from eq.data.collector import collect_hk_daily
    codes_list = [c.strip().zfill(5) for c in codes.split(",") if c.strip()] if codes else None
    collect_hk_daily(top_n=top_n, start=start, codes=codes_list)


@data_app.command("hk-5min", help="港股 5 分钟线（yfinance，最近 30 天）")
def data_hk_5min(
    top_n: int = typer.Option(200, "--top", "-n", help="前 N 只"),
    codes_file: str = typer.Option("", "--codes-file", help="自定义品种表 txt，自动解析其中港股代码"),
):
    from eq.data.collector import collect_hk_minute
    from eq.data.hk_market import parse_hk_codes_from_file
    codes = parse_hk_codes_from_file(codes_file) if codes_file else None
    collect_hk_minute(codes=codes, top_n=top_n, interval="5m", period="1mo")


@data_app.command("hk-1min", help="港股 1 分钟线（yfinance，最近 7 天）")
def data_hk_1min(
    top_n: int = typer.Option(200, "--top", "-n", help="前 N 只"),
    codes_file: str = typer.Option("", "--codes-file", help="自定义品种表 txt，自动解析其中港股代码"),
):
    from eq.data.collector import collect_hk_minute
    from eq.data.hk_market import parse_hk_codes_from_file
    codes = parse_hk_codes_from_file(codes_file) if codes_file else None
    collect_hk_minute(codes=codes, top_n=top_n, interval="1m", period="7d")


@data_app.command("us", help="美股日线（yfinance）")
def data_us(
    top_n: int = typer.Option(100, "--top", "-n", help="前 N 只"),
    start: str = typer.Option("2024-01-01", "--start", "-s", help="起始日"),
    codes: str = typer.Option("", "--codes", help="显式指定美股代码（逗号分隔，如 AAPL,MSFT），优先于 top"),
):
    from eq.data.collector import collect_us_daily
    codes_list = [c.strip().upper() for c in codes.split(",") if c.strip()] if codes else None
    collect_us_daily(top_n=top_n, start=start, codes=codes_list)


@data_app.command("all", help="全量数据收集（A股+港股日线+5min+1min+美股）")
def data_all(
    top_n: int = typer.Option(200, "--top", "-n", help="港股/美股前 N 只"),
):
    from eq.data.collector import collect_a_share, collect_hk_daily, collect_hk_minute, collect_us_daily
    print("\n===== 全量数据收集 =====\n")
    collect_a_share()
    collect_hk_daily(top_n=top_n)
    collect_hk_minute(top_n=top_n, interval="5m", period="2mo")
    collect_hk_minute(top_n=top_n, interval="1m", period="7d")
    collect_us_daily(top_n=top_n)
    print("\n===== 全部完成 =====\n")


@data_app.command("migrate", help="把旧散落目录（.qlib_data、hk_data、us_data）的数据迁移到统一 data/ 目录")
def data_migrate(
    dry_run: bool = typer.Option(False, "--dry-run", help="只打印将做什么，不真正复制"),
):
    from eq.data.paths import migrate_legacy_data_layout
    result = migrate_legacy_data_layout(dry_run=dry_run, verbose=True)
    typer.echo(f"\n迁移完成：复制 {len(result['copied'])} 项，跳过 {len(result['skipped'])} 项")
    typer.echo("旧目录保留，可手动删除：.qlib_data/、.eternityquant/hk_data/、.eternityquant/us_data/")


@data_app.command("sources", help="列出全部数据源；--test 在本机实测哪些真的通（结果影响后续取数优先级）")
def data_sources(
    test: bool = typer.Option(False, "--test", "-t", help="实际打网络测一遍（每个源几秒）"),
    market: str = typer.Option("", "--market", "-m", help="只测某市场：A|HK|US|CRYPTO"),
    cap: str = typer.Option("", "--cap", "-c", help="只测某能力：bars|snapshot|spot"),
    workers: int = typer.Option(6, "--workers", "-w", help="并发数"),
):
    from eq.data import sources as sr

    if not test:
        df = sr.describe_registry()
        print(f"\n已注册数据源（共 {len(df)} 个）：\n")
        print(df.to_string(index=False))
        health = sr.load_health()
        if health:
            typer.echo(f"\n上次自检：{health.get('tested_at', '?')}（{health.get('n_jobs', 0)} 项）"
                       f"　用 `eq data sources --test` 重测")
        else:
            typer.echo("\n还没跑过自检。跑一次 `eq data sources --test`，"
                       "之后取数会自动优先用你这台机器上真正通的源。")
        return

    markets = [market.upper()] if market else None
    caps = [cap.lower()] if cap else None
    if markets and markets[0] not in ("A", "HK", "US", "CRYPTO"):
        typer.echo(f"未知市场 {market}，可选 A/HK/US/CRYPTO", err=True)
        raise typer.Exit(1)
    if caps and caps[0] not in ("bars", "snapshot", "spot"):
        typer.echo(f"未知能力 {cap}，可选 bars/snapshot/spot", err=True)
        raise typer.Exit(1)

    typer.echo("在本机实测各数据源（结果会存下来，之后取数自动优先用通的源）...\n")
    print(f"{'源':<12} {'市场':<7} {'能力':<9} {'状态':<5} {'耗时':>7}  详情")
    print("-" * 108)

    def _show(name, m, c, res):
        mark = "✓" if res["ok"] else "✗"
        print(f"{name:<12} {m:<7} {c:<9} {mark:<5} {res['seconds']:>6.2f}s  {res['detail'][:56]}",
              flush=True)

    report = sr.probe_all(markets=markets, caps=caps, workers=workers, on_result=_show)

    # 汇总：每个市场哪些源可用
    print("\n" + "=" * 60)
    results = report["results"]
    for m in (markets or ["A", "HK", "US", "CRYPTO"]):
        ok = [n for n, mk in results.items() if mk.get(m, {}).get("ok")]
        bad = [n for n, mk in results.items() if m in mk and not mk[m].get("ok")]
        if not ok and not bad:
            continue
        typer.echo(f"\n{m} 市场：")
        typer.echo(f"  ✓ 可用（{len(ok)}）：{', '.join(ok) or '无'}")
        if bad:
            typer.echo(f"  ✗ 不通（{len(bad)}）：{', '.join(bad)}")
        if not ok:
            typer.echo("  ⚠ 该市场无可用源，取数会直接失败")
    typer.echo(f"\n自检结果已存到 {sr._health_path()}")
    typer.echo("后续 eq watch / scan / backtest / screen 会自动优先用实测通的源。")


@data_app.command("paths", help="显示统一数据目录结构")
def data_paths():
    from eq.data.paths import (
        QLIB_CN_DATA_DIR,
        HK_DIR, HK_DAILY_DIR, HK_5M_DIR, HK_1M_DIR, HK_FEAT_DIR, HK_MODELS_DIR,
        US_DIR, US_DAILY_DIR,
    )
    typer.echo("\n统一数据目录结构：\n")
    for label, d in [
        ("A 股 qlib", QLIB_CN_DATA_DIR),
        ("港股 根目录", HK_DIR),
        ("  日线", HK_DAILY_DIR),
        ("  5 分钟", HK_5M_DIR),
        ("  1 分钟", HK_1M_DIR),
        ("  特征", HK_FEAT_DIR),
        ("  模型", HK_MODELS_DIR),
        ("美股 根目录", US_DIR),
        ("  日线", US_DAILY_DIR),
    ]:
        n = len(list(d.iterdir())) if d.exists() else "不存在"
        typer.echo(f"  {label:16s} {n:>4} 项  {d}")


@app.command("theme", help="配置仪表盘主题（看板娘图片 + 自动配色），不带参数看当前设置")
def theme_cmd(
    image: str = typer.Argument("", help="图片路径；留空则显示当前配置"),
    opacity: float = typer.Option(-1.0, "--opacity", "-o",
                                  help="背景遮罩不透明度 0~1，越大背景越淡（默认 0.88）"),
    mascot: bool = typer.Option(True, "--mascot/--no-mascot", help="侧边栏看板娘"),
    primary: str = typer.Option("", "--primary", help="手动指定主色 #rrggbb，缺省从图片自动提取"),
    clear: bool = typer.Option(False, "--clear", help="清除主题，恢复默认外观"),
):
    from eq.db import DEFAULT_HOME
    from eq.web.theme import extract_palette, load_ui_config, save_image_to_env

    env_file = DEFAULT_HOME / ".env"

    def _set(key: str, value: str | None) -> None:
        """写入/删除 .env 里的一行。"""
        env_file.parent.mkdir(parents=True, exist_ok=True)
        lines = ([ln for ln in env_file.read_text(encoding="utf-8").splitlines()
                  if not ln.strip().startswith(f"{key}=")]
                 if env_file.exists() else [])
        if value is not None:
            lines.append(f"{key}={value}")
        env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if clear:
        for k in ("EQ_DASH_IMAGE", "EQ_DASH_OPACITY", "EQ_DASH_MASCOT", "EQ_DASH_PRIMARY"):
            _set(k, None)
        typer.echo("已清除主题配置，仪表盘恢复默认外观")
        return

    if image:
        try:
            save_image_to_env(image)
        except FileNotFoundError as e:
            typer.echo(f"{e}", err=True)
            raise typer.Exit(1) from e
    if opacity >= 0:
        _set("EQ_DASH_OPACITY", str(min(opacity, 1.0)))
    if not mascot:
        _set("EQ_DASH_MASCOT", "off")
    elif image:
        _set("EQ_DASH_MASCOT", "on")
    if primary:
        from eq.web.theme import _valid_hex

        if not _valid_hex(primary):
            typer.echo(f"主色格式不对：{primary}（应为 #rrggbb，如 #b08968）", err=True)
            raise typer.Exit(1)
        _set("EQ_DASH_PRIMARY", primary)

    # 重新读一次，展示最终生效的配置
    import os as _os
    for k in ("EQ_DASH_IMAGE", "EQ_DASH_OPACITY", "EQ_DASH_MASCOT", "EQ_DASH_PRIMARY"):
        _os.environ.pop(k, None)     # 清掉进程内的旧值，强制从文件重读
    cfg = load_ui_config()
    if not cfg.enabled:
        typer.echo("当前未配置主题（仪表盘为默认外观）。")
        typer.echo("  设置：eq theme \"D:\\图片\\your.jpg\"")
        return
    typer.echo(f"\n主题配置（{env_file}）")
    typer.echo(f"  图片      {cfg.image}")
    typer.echo(f"  遮罩      {cfg.opacity}（越大背景越淡）")
    typer.echo(f"  看板娘    {'开' if cfg.mascot else '关'}")
    pal = extract_palette(cfg.image)
    typer.echo(f"  自动配色  主色 {cfg.primary or pal['accent']}   "
               f"背景 {pal['overlay']}   {'亮色' if pal['is_light'] else '暗色'}主题")
    typer.echo("\n用 `eq dash` 启动看效果；不满意可调 --opacity 或换图。")


@app.command("dash", help="启动 Streamlit 仪表盘（本地网页看板，支持看板娘主题）")
def dash(
    port: int = typer.Option(8501, "--port", "-p", help="本地端口"),
    image: str = typer.Option("", "--image", "-i",
                              help="主题图片路径（背景 + 侧边栏看板娘，主题色自动从图提取）"),
    opacity: float = typer.Option(-1.0, "--opacity",
                                  help="背景遮罩不透明度 0~1，越大背景越淡；-1=用配置/默认"),
    save: bool = typer.Option(False, "--save",
                              help="把 --image 写进 .eternityquant/.env，以后不用再传"),
    no_theme: bool = typer.Option(False, "--no-theme", help="本次禁用主题（排查显示问题用）"),
):
    import os as _os

    from eq.web import run_dashboard

    if image:
        from eq.web.theme import save_image_to_env

        p = image.strip().strip('"').strip("'")
        if not __import__("pathlib").Path(p).exists():
            typer.echo(f"图片不存在：{p}", err=True)
            raise typer.Exit(1)
        _os.environ["EQ_DASH_IMAGE"] = p
        if save:
            env_file = save_image_to_env(p)
            typer.echo(f"主题图片已写入 {env_file}（EQ_DASH_IMAGE），以后 eq dash 直接生效")
    if opacity >= 0:
        _os.environ["EQ_DASH_OPACITY"] = str(min(opacity, 1.0))
    if no_theme:
        _os.environ["EQ_DASH_IMAGE"] = ""

    code = run_dashboard(port=port)
    if code != 0:
        typer.echo(f"Streamlit 异常退出：{code}", err=True)
        raise typer.Exit(code)


# ---------- 入口（必须放在文件最末尾：所有子命令注册完之后） ----------

if __name__ == "__main__":
    app()
