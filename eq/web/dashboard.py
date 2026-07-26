"""Streamlit 仪表盘主入口（被 streamlit run 直接执行）。

侧边栏分页：
- 概览（持仓 + 自选 + 最新信号汇总）
- 持仓
- 自选
- 监控规则
- ML 模型（v0.11：详情 + 激活 + 批量预测 Top10 可交互）
- 回测
- 深度研究（v0.11：输入 symbol → 14 板块深度研究）
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from eq.core import monitor as mon_svc
from eq.core import portfolio as pf_svc
from eq.core import watchlist as wl_svc
from eq.db import execute
from eq.strategy.factors import ml as ml_svc

st.set_page_config(page_title="EternityQuant", page_icon="📊", layout="wide")

# v0.33：主题定制（看板娘 + 从图片自动配色）。配置在 .eternityquant/.env：
#   EQ_DASH_IMAGE=D:\path\to\your.jpg   EQ_DASH_OPACITY=0.88   EQ_DASH_MASCOT=on
# 没配置就保持默认外观；主题任何一步失败都静默降级，不拖垮仪表盘。
from eq.web import theme as _theme  # noqa: E402

_theme_info = _theme.apply(st)


def _fmt_df(rows: list[dict]) -> pd.DataFrame:
    """列表字典转 DataFrame 显示。"""
    return pd.DataFrame(rows) if rows else pd.DataFrame()


st.title("EternityQuant 个人散户量化助手")
st.caption(f"今日 {dt.date.today().isoformat()}")

page = st.sidebar.selectbox(
    "页面",
    ["概览", "晨报", "持仓", "自选", "选股", "回测", "监控规则", "ML 模型",
     "下载管理", "深度研究"],
    index=0,
)

# -------- 概览 --------
if page == "概览":
    st.header("概览")
    _rules = mon_svc.list_rules()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("自选股数", len(wl_svc.list_all()))
    col2.metric("持仓只数", len(pf_svc.list_open()))
    col3.metric("监控规则", f"{sum(1 for r in _rules if r['enabled'])}/{len(_rules)}",
                help="启用中 / 总数")
    try:
        from eq.data.cache import stats as _cache_stats
        _cs = _cache_stats()
        col4.metric("行情缓存", f"{_cs['symbols']} 只", f"{_cs['size_mb']} MB")
    except Exception:
        col4.metric("行情缓存", "-")

    st.subheader("当前持仓")
    if not pf_svc.list_open():
        st.info("无持仓")
    else:
        with st.spinner("拉最新行情中..."):
            _s = pf_svc.summary()
        m1, m2, m3 = st.columns(3)
        m1.metric("总市值", f"{_s['total_market_value']:,.0f}")
        m2.metric("浮动盈亏", f"{_s['total_unrealized_pnl']:+,.0f}", f"{_s['total_unrealized_pct']:+.2f}%")
        m3.metric("今日盈亏", f"{_s['total_today_pnl']:+,.0f}")
        st.dataframe(pd.DataFrame([
            {"符号": p["symbol"], "股数": p["shares"], "成本": round(p["cost_price"], 2),
             "现价": round(p["current_price"], 2), "浮盈": round(p["unrealized_pnl"], 2),
             "浮盈%": round(p["unrealized_pct"], 2), "今日%": round(p["today_pct"], 2),
             "占比%": round(p["weight_pct"], 1)}
            for p in _s["positions"]
        ]), width="stretch")

    st.subheader("最近触发信号")
    sigs = mon_svc.recent_signals(limit=20)
    if not sigs:
        st.info("尚无规则触发记录（规则触发后会自动落 signals 表）")
    else:
        st.dataframe(pd.DataFrame([
            {"时间": str(s["created_at"])[:19], "标的": s["symbol"],
             "类型": s["signal_type"], "标题": (s.get("context") or {}).get("title", "")}
            for s in sigs
        ]), width="stretch")

# -------- 晨报（v0.33：CLI 的 eq daily / eq paper 搬进看板） --------
elif page == "晨报":
    st.header("每日晨报")
    st.caption("大盘闸门 → 持仓止损 → 今日信号翻转 → 纸面战绩。等同命令行的 `eq daily`。")

    from eq.core import briefing as _brf
    from eq.core import journal as _jnl
    from eq.strategy.registry import list_strategies as _list_strats
    from eq.strategy.registry import resolve as _resolve_strategy

    _names = _list_strats()
    _c1, _c2, _c3 = st.columns([2, 2, 1])
    _strat = _c1.selectbox("信号策略", _names,
                           index=_names.index("trend_vote") if "trend_vote" in _names else 0)
    _bench = _c2.text_input("基准指数", "000300.SH")
    # 默认不写库：看板是随手点开的，误点一下就往纸面日志里灌重复推荐不合适。
    # 命令行 eq daily 才是"每天跑一次"的正规入口，那边默认记录。
    _rec = _c3.checkbox("记入纸面", value=False, help="把今日买入信号写进 paper_recos")

    if st.button("生成晨报", type="primary"):
        from concurrent.futures import ThreadPoolExecutor

        from eq.data.market import get_recent_bars
        _fn = _resolve_strategy(_strat)

        with st.spinner("拉大盘…"):
            try:
                _idx = get_recent_bars(_bench, days=500)
            except Exception as e:
                _idx = None
                st.warning(f"基准 {_bench} 拉取失败：{e}")
        _ms = _brf.market_status(_idx)
        st.subheader("大盘")
        if _ms:
            g1, g2, g3 = st.columns(3)
            g1.metric(_bench, f"{_ms['close']:.2f}", f"{_ms['change_pct']:+.2f}%")
            g2.metric("距 MA200",
                      f"{_ms['dist_ma_pct']:+.1f}%" if _ms["dist_ma_pct"] is not None else "-")
            g3.metric("闸门", "开" if _ms["gate_open"] else "关",
                      help="收盘在长期均线之上才允许持股，主要作用是压回撤")
            if not _ms["gate_open"]:
                st.warning("大盘闸门关闭——买入信号谨慎对待，优先控回撤")
        else:
            st.info("拿不到基准行情，跳过大盘判断")

        st.subheader("持仓止损")
        _pos = pf_svc.list_open()
        if not _pos:
            st.info("空仓")
        else:
            try:
                _sm = pf_svc.summary()
                _chk = _brf.stop_breaches(_sm["positions"])
                for p in _chk["breached"]:
                    st.error(f"‼ {p['symbol']} 已跌破止损（现价 {p['current_price']:.2f}"
                             f" ≤ 止损 {p['stop_loss']:.2f}）——按纪律该走了")
                for p in _chk["near"]:
                    _d = (p["current_price"] - p["stop_loss"]) / p["current_price"]
                    st.warning(f"⚠ {p['symbol']} 逼近止损（距 {_d:.1%}）")
                if _chk["no_stop"]:
                    st.info("未设止损：" + "、".join(p["symbol"] for p in _chk["no_stop"][:8])
                            + "（可在「持仓」页补）")
                if not (_chk["breached"] or _chk["near"] or _chk["no_stop"]):
                    st.success("全部持仓止损安全")
            except Exception as e:
                st.error(f"体检失败：{e}")

        st.subheader("今日信号翻转")
        _syms = sorted({w["symbol"] for w in wl_svc.list_all()}
                       | {p["symbol"] for p in _pos})
        if not _syms:
            st.info("候选池为空，先去「自选」页加几只")
        else:
            def _pull(x):
                try:
                    return x, get_recent_bars(x, days=400)
                except Exception:
                    return x, None

            with st.spinner(f"拉 {len(_syms)} 只行情…"):
                with ThreadPoolExecutor(max_workers=min(8, len(_syms))) as _pool:
                    _bars = {x: d for x, d in _pool.map(_pull, _syms)
                             if d is not None and len(d) >= 30}
            _chg = _brf.detect_signal_changes(_bars, _fn)
            _en = sorted(k for k, v in _chg.items() if v == "enter")
            _ex = sorted(k for k, v in _chg.items() if v == "exit")
            s1, s2, s3 = st.columns(3)
            s1.metric("今日新买入", len(_en))
            s2.metric("今日转卖出", len(_ex))
            s3.metric("持有中", sum(1 for v in _chg.values() if v == "holding"))
            if _en:
                st.success("买入：" + "、".join(_en))
            if _ex:
                st.error("卖出：" + "、".join(_ex))
            if not _en and not _ex:
                st.info("今日无翻转——不动就是最好的操作")
            if _rec and _chg:
                _recos, _bpx = _brf.build_recos(_bars, _chg, _idx)
                if _recos:
                    _n = _jnl.record(_recos, _strat, horizon_days=10,
                                     benchmark=_bench, benchmark_price=_bpx)
                    st.caption(f"已记入纸面日志 {_n} 笔（10 交易日后自动结算）")

    st.divider()
    st.subheader("纸面战绩")
    st.caption("前向记录、样本外结算，是这套系统里唯一没法事后调参的验证。")
    if st.button("结算到期推荐"):
        try:
            _st = _jnl.evaluate_due()
            st.success(f"结算 {len(_st)} 笔") if _st else st.info("没有到期的推荐")
        except Exception as e:
            st.error(f"结算失败：{e}")
    _sb = _jnl.scoreboard()
    if not _sb.get("n_closed"):
        st.info(f"尚无已结算记录（在途 {_sb.get('n_open', 0)} 笔）。"
                "跑 `eq daily` 攒几周再看。")
    else:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("已结算", _sb["n_closed"], f"在途 {_sb['n_open']}")
        p2.metric("平均收益", f"{_sb['ret_mean']:+.2%}")
        p3.metric("胜率", f"{_sb['win_rate']:.0%}")
        if _sb.get("excess_t") is not None:
            p4.metric("超额 t 值", f"{_sb['excess_t']:+.2f}",
                      help="|t| ≥ 2 才算统计显著；否则和运气分不开")
            if abs(_sb["excess_t"]) < 2 and _sb.get("n_for_significance"):
                st.caption(f"按当前均值/波动，还需约 {_sb['n_for_significance']} 笔才能到 t=2")
        _rc = _jnl.recent_closed(limit=20)
        if _rc:
            st.dataframe(pd.DataFrame([
                {"推荐日": r["reco_date"], "标的": r["symbol"], "策略": r["strategy"],
                 "买入": round(r["entry_price"], 2), "卖出": round(r["exit_price"], 2),
                 "收益%": round(r["ret"] * 100, 2),
                 "超额%": round(r["excess"] * 100, 2) if r["excess"] is not None else None}
                for r in _rc
            ]), width="stretch")

# -------- 持仓 --------
elif page == "持仓":
    st.header("当前持仓")
    if not pf_svc.list_open():
        st.info("无持仓")
    else:
        with st.spinner("拉最新行情算浮盈中..."):
            s = pf_svc.summary()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总市值", f"{s['total_market_value']:,.0f}")
        c2.metric("浮动盈亏", f"{s['total_unrealized_pnl']:+,.0f}",
                  f"{s['total_unrealized_pct']:+.2f}%")
        c3.metric("今日盈亏", f"{s['total_today_pnl']:+,.0f}")
        c4.metric("累计已实现", f"{s['total_realized_pnl']:+,.0f}")

        st.dataframe(pd.DataFrame(s["positions"]), width="stretch")

        # 风险提示
        risks = []
        if s["max_weight_pct"] > 30:
            risks.append(f"集中度偏高：**{s['max_weight_symbol']}** 占 {s['max_weight_pct']:.1f}%")
        if s["no_stop"]:
            risks.append(f"未设止损 {len(s['no_stop'])} 只：{', '.join(s['no_stop'][:8])}")
        if s["stale"]:
            risks.append(f"行情拉取失败 {len(s['stale'])} 只（用成本价占位）：{', '.join(s['stale'][:8])}")
        if risks:
            st.warning("风险提示\n\n" + "\n\n".join(f"- {r}" for r in risks))

        # 仓位分布
        if s["total_market_value"] > 0:
            st.subheader("仓位分布")
            weights = pd.DataFrame(
                [{"symbol": p["symbol"], "占比%": round(p["weight_pct"], 2)} for p in s["positions"]]
            ).set_index("symbol")
            st.bar_chart(weights)

    st.subheader("已清仓记录")
    df_closed = _fmt_df(pf_svc.list_closed(limit=50))
    if df_closed.empty:
        st.info("无已清仓记录")
    else:
        st.dataframe(df_closed, width="stretch")
        st.metric("已清仓累计实现盈亏", f"{df_closed['realized_pnl'].sum():+,.2f}")

# -------- 自选 --------
elif page == "自选":
    st.header("自选股")
    rows = wl_svc.list_all()
    if not rows:
        st.info("自选列表为空")
    else:
        st.caption(f"共 {len(rows)} 只")
        if st.button("📈 拉取实时行情"):
            with st.spinner("并发拉取中..."):
                st.session_state["wl_quotes"] = wl_svc.quotes()
        if st.session_state.get("wl_quotes"):
            st.dataframe(pd.DataFrame(st.session_state["wl_quotes"]), width="stretch")
        else:
            st.dataframe(pd.DataFrame(rows), width="stretch")

        with st.expander("➕ 加入自选"):
            c1, c2, c3 = st.columns([2, 2, 2])
            new_sym = c1.text_input("代码", key="wl_new_sym")
            new_tags = c2.text_input("标签（逗号分隔）", key="wl_new_tags")
            new_reason = c3.text_input("理由", key="wl_new_reason")
            if st.button("加入", key="wl_add_btn") and new_sym.strip():
                wl_svc.add(new_sym.strip(), reason=new_reason, tags=new_tags)
                st.success(f"已加入 {new_sym}")
                st.rerun()

# -------- 选股 --------
elif page == "选股":
    from eq.core.screener import CONDITIONS, screen as do_screen

    st.header("技术选股")
    st.caption("对自选/持仓/市场榜跑技术条件筛选，命中的可一键加入自选")
    c1, c2, c3 = st.columns([2, 1, 1])
    conds = c1.multiselect(
        "筛选条件", list(CONDITIONS.keys()),
        default=["golden_cross"],
        format_func=lambda k: f"{k}（{CONDITIONS[k][1]}）",
    )
    src = c2.selectbox("候选池", ["watchlist", "portfolio", "A", "HK", "US"], index=0)
    mode = c3.selectbox("组合方式", ["all", "any"], index=0,
                        format_func=lambda m: "全部满足" if m == "all" else "任一满足")
    c4, c5 = st.columns(2)
    pool_top = c4.number_input("市场榜候选数", min_value=10, max_value=500, value=100, step=10)
    bars = c5.number_input("每只拉多少根日线", min_value=30, max_value=750, value=120, step=10)

    if st.button("🔍 开始筛选", type="primary") and conds:
        if src == "watchlist":
            symbols = [r["symbol"] for r in wl_svc.list_all()]
        elif src == "portfolio":
            symbols = [r["symbol"] for r in pf_svc.list_open()]
        else:
            from eq.core.scanner import scan as _scan
            symbols = _scan(src, sort_by="amount", top_n=int(pool_top))["symbol"].astype(str).tolist()
        if not symbols:
            st.warning("候选池为空")
        else:
            with st.spinner(f"筛选 {len(symbols)} 只标的中..."):
                try:
                    hits = do_screen(symbols, conds, mode=mode, days=int(bars))
                except Exception as e:
                    hits = None
                    st.error(f"筛选失败：{repr(e)[:300]}")
            if hits is not None:
                if not hits:
                    st.info(f"{len(symbols)} 只候选中无标的命中")
                else:
                    st.success(f"命中 {len(hits)} 只")
                    st.session_state["screen_hits"] = hits
    if st.session_state.get("screen_hits"):
        hits = st.session_state["screen_hits"]
        st.dataframe(pd.DataFrame([
            {"symbol": h["symbol"], "最新价": round(h["close"], 2),
             "涨跌幅%": round(h["change_pct"], 2), "命中数": h["score"],
             "命中条件": ", ".join(h["matched"]), "原因": "; ".join(h["reasons"])}
            for h in hits
        ]), width="stretch")
        tag = st.text_input("加入自选的标签", "screen", key="screen_tag")
        if st.button("➕ 把命中结果加入自选"):
            n = sum(1 for h in hits if wl_svc.add(
                h["symbol"], reason=f"screen: {','.join(h['matched'])}", tags=tag))
            st.success(f"已加入 {n} 只新标的")

# -------- 回测 --------
elif page == "回测":
    from eq.backtest import BacktestConfig, EventDrivenBacktester, VectorizedBacktester
    from eq.backtest.store import list_runs, load_result, remove_run
    # v0.33：直接用共享注册表（原来这里手抄了 4 个，新加的十几个策略网页上看不到）
    from eq.strategy.registry import builtin_strategies

    _STRATS = builtin_strategies()

    st.header("回测")
    tab_run, tab_hist = st.tabs(["跑回测", "历史记录"])

    with tab_run:
        c1, c2, c3 = st.columns([2, 2, 1])
        bt_sym = c1.text_input("标的", "600519.SH", key="bt_sym")
        bt_strat = c2.selectbox("策略", ["（全部横评）", *sorted(_STRATS)], index=0)
        bt_engine = c3.selectbox("引擎", ["vectorized", "event_driven"], index=0)
        c4, c5, c6, c7 = st.columns(4)
        bt_days = c4.number_input("回测天数", min_value=30, max_value=2000, value=365, step=30)
        bt_cash = c5.number_input("初始现金", min_value=10000, value=1000000, step=10000)
        bt_comm = c6.number_input("手续费(bps)", min_value=0.0, value=2.5, step=0.5)
        bt_slip = c7.number_input("滑点(bps)", min_value=0.0, value=5.0, step=0.5)

        if st.button("▶ 开始回测", type="primary") and bt_sym.strip():
            from eq.data.market import get_recent_bars
            with st.spinner("拉行情 + 回测中..."):
                try:
                    bars_df = get_recent_bars(bt_sym.strip(), days=int(bt_days))
                except Exception as e:
                    bars_df = None
                    st.error(f"拉行情失败：{repr(e)[:300]}")
            if bars_df is not None and not bars_df.empty:
                def _mk_engine():
                    return VectorizedBacktester() if bt_engine == "vectorized" else EventDrivenBacktester()

                names = sorted(_STRATS) if bt_strat == "（全部横评）" else [bt_strat]
                out_rows, results = [], {}
                for nm in names:
                    cfg = BacktestConfig(initial_cash=float(bt_cash), commission_bps=float(bt_comm),
                                         slippage_bps=float(bt_slip), engine=bt_engine)
                    try:
                        res = _mk_engine().run(bars_df, _STRATS[nm], cfg)
                    except Exception as e:
                        st.error(f"{nm} 回测失败：{repr(e)[:200]}")
                        continue
                    results[nm] = res
                    m = res.metrics
                    out_rows.append({
                        "策略": nm, "总收益%": round(m["total_return"] * 100, 2),
                        "年化%": round(m["annual_return"] * 100, 2), "夏普": round(m["sharpe"], 2),
                        "Sortino": round(m["sortino"], 2), "最大回撤%": round(m["max_drawdown"] * 100, 2),
                        "胜率%": round(m["win_rate"] * 100, 1), "交易": m["num_trades"],
                    })
                if out_rows:
                    st.dataframe(pd.DataFrame(out_rows).sort_values("夏普", ascending=False),
                                 width="stretch")
                    curves = pd.DataFrame({nm: r.equity_curve for nm, r in results.items()})
                    curves["买入持有"] = float(bt_cash) * bars_df["close"] / float(bars_df["close"].iloc[0])
                    st.subheader("权益曲线（含买入持有基准）")
                    st.line_chart(curves)
                    if st.checkbox("保存这批回测到历史记录"):
                        from eq.backtest.store import save_result
                        for nm, r in results.items():
                            save_result(r, symbol=bt_sym.strip(), strategy_name=nm)
                        st.success(f"已保存 {len(results)} 条")

    with tab_hist:
        runs = list_runs(limit=100)
        if not runs:
            st.info("暂无回测记录")
        else:
            hist_rows = []
            for r in runs:
                m = r.get("metrics") or {}
                hist_rows.append({
                    "run_id": r["id"], "标的": r["symbol"], "策略": r["strategy_name"],
                    "引擎": r["engine"], "总收益%": round(m.get("total_return", 0) * 100, 2),
                    "夏普": round(m.get("sharpe", 0), 2),
                    "最大回撤%": round(m.get("max_drawdown", 0) * 100, 2),
                    "时间": str(r["created_at"])[:19],
                })
            st.dataframe(pd.DataFrame(hist_rows), width="stretch")
            sel = st.selectbox("查看详情", [r["id"] for r in runs])
            cc1, cc2 = st.columns([1, 1])
            if cc1.button("📊 加载详情"):
                try:
                    bundle = load_result(sel)
                    st.json(bundle["meta"]["metrics"])
                    if not bundle["equity"].empty:
                        st.line_chart(bundle["equity"])
                    if not bundle["trades"].empty:
                        st.dataframe(bundle["trades"], width="stretch")
                except Exception as e:
                    st.error(f"加载失败：{repr(e)[:200]}")
            if cc2.button("🗑 删除该记录"):
                st.success("已删除" if remove_run(sel) else "记录不存在")
                st.rerun()

# -------- 监控规则 --------
elif page == "监控规则":
    st.header("监控规则")
    rules = mon_svc.list_rules()
    if not rules:
        st.info("无监控规则")
    else:
        df = pd.DataFrame(rules)
        st.dataframe(df, width="stretch")
        st.subheader("触发统计")
        fired = [r for r in rules if r["fire_count"] > 0]
        st.metric("累计触发条数", sum(r["fire_count"] for r in rules))
        if fired:
            st.dataframe(_fmt_df(fired), width="stretch")

# -------- ML 模型（v0.11：详情 + 激活 + 批量预测 Top10） --------
elif page == "ML 模型":
    st.header("ML 模型")
    models = ml_svc.list_models()
    if not models:
        st.info("无模型记录，用 `eq ml train` 训练后再来")
    else:
        df = pd.DataFrame(models)
        # 模型列表
        st.subheader("全部模型")
        st.dataframe(df, width="stretch")

        # 当前激活模型
        active = [m for m in models if m.get("is_active") == 1]
        st.subheader("当前激活模型")
        if not active:
            st.warning("无激活模型")
        else:
            st.dataframe(pd.DataFrame(active), width="stretch")

        # 激活操作
        st.subheader("激活/切换模型")
        model_opts = {f"{m['id']}  {m['name']}  IC={m.get('metrics',{}).get('ic',0):+.4f}": m["id"] for m in models}
        chosen = st.selectbox("选模型", list(model_opts.keys()))
        if st.button("激活"):
            ml_svc.activate(model_opts[chosen])
            st.success(f"已激活 {chosen}")
            st.rerun()

        # 批量预测 Top10
        st.subheader("批量预测 Top10")
        active_id = active[0]["id"] if active else None
        if active_id is None:
            st.info("先激活一个模型才能批量预测")
        else:
            st.caption(f"用激活模型 {active_id} 跑全 universe 批量预测")
            if st.button("跑 predict-batch"):
                with st.spinner("qlib init + 加载模型 + 跑预测...（约 1-2 分钟）"):
                    try:
                        from eq.strategy.factors.ml_workflow import predict_batch
                        pred_df = predict_batch(active_id, top_n=10)
                        if pred_df.empty:
                            st.warning("预测结果为空")
                        else:
                            st.success("Top10 预测完成（已写入 ml_predictions 表）")
                            st.dataframe(pred_df, width="stretch")
                            # 收藏到自选股的快捷入口
                            if st.checkbox("把 Top10 加入自选股"):
                                for _, row in pred_df.iterrows():
                                    try:
                                        wl_svc.add(row["symbol"], reason=f"ML Top10 score={row['score']:+.4f}", tags="ml,top10")
                                    except Exception:
                                        pass
                                st.success(f"{len(pred_df)} 只已加入自选股")
                    except Exception as e:
                        st.error(f"预测失败：{repr(e)[:300]}")

        # 某模型的预测历史
        st.subheader("某模型的预测历史")
        hist_model = st.selectbox("选模型查预测", [m["id"] for m in models], index=0)
        rows = execute("SELECT symbol, date, score FROM ml_predictions WHERE model_id = ? ORDER BY date DESC LIMIT 50", (hist_model,))
        if not rows:
            st.info("该模型暂无预测记录")
        else:
            # sqlite3.Row 是序列不是映射，pd.DataFrame(rows) 会得到 0/1/2 数字列名，
            # 列名全丢。先显式转 dict 再建 DataFrame。
            st.dataframe(pd.DataFrame([dict(r) for r in rows]), width="stretch")

# -------- 深度研究（v0.11） --------
elif page == "深度研究":
    st.header("个股深度研究")
    st.caption("按市场自动选数据源汇总：A 股 11 板块 / 港 4 / 美 6 / 加密 1；港美拉取失败会显 MCP 补全建议")
    sym = st.text_input("股票符号", placeholder="如 600519.SH / AAPL.US / 00700.HK / BTC-USDT")
    # 板块选择
    from eq.core.research import _DEFAULT_SECTIONS, _SECTION_LABELS
    preset_secs = list(_DEFAULT_SECTIONS.get("A", ["snapshot"]))
    secs_chosen = st.multiselect("板块（缺省按市场全拉）", list(_SECTION_LABELS.keys()), default=[], format_func=lambda s: _SECTION_LABELS.get(s, s))
    if st.button("跑深度研究") and sym:
        from eq.core.research import format_research, research as do_research
        with st.spinner("拉取数据中..."):
            try:
                report = do_research(sym, sections=secs_chosen or None)
                # 文本版
                st.code(format_research(report), language="text")
                # 结构化展开
                st.subheader("结构化展开")
                for sec, data in report.items():
                    if sec in ("symbol", "market", "as_of"):
                        continue
                    with st.expander(f"{_SECTION_LABELS.get(sec, sec)}"):
                        if isinstance(data, dict) and "error" in data:
                            st.error(data["error"])
                        elif isinstance(data, dict) and "hint" in data:
                            st.info(f"💡 {data['hint']}")
                        elif isinstance(data, dict) and "snapshot" in data:
                            snap = data["snapshot"]
                            cols = st.columns(4)
                            cols[0].metric("最新价", snap.get("close", "?"))
                            cols[1].metric("漲跌幅", f"{snap.get('change_pct', 0):+.2f}%")
                            cols[2].metric("今开", snap.get("open", "?"))
                            cols[3].metric("昨收", snap.get("prev_close", "?"))
                            if data.get("recent_30d"):
                                r = data["recent_30d"]
                                st.caption(f"近 {r['days']} 日：高 {r['high']:.2f}  低 {r['low']:.2f}  均价 {r['avg_close']:.2f}")
                        elif isinstance(data, dict) and "info" in data:
                            st.json(data["info"])
                        elif isinstance(data, dict) and "headlines" in data:
                            for h in data["headlines"][:10]:
                                if isinstance(h, dict):
                                    title = h.get("新闻标题") or h.get("title") or str(h)[:80]
                                    st.text(f"• {title}")
                        elif isinstance(data, dict) and "reports" in data:
                            for r in data["reports"][:10]:
                                if isinstance(r, dict):
                                    title = r.get("研报标题") or r.get("title") or str(r)[:80]
                                    st.text(f"• {title}")
                        elif isinstance(data, dict) and "recent_5d" in data:
                            st.dataframe(pd.DataFrame(data["recent_5d"]), width="stretch")
                        elif isinstance(data, dict) and "recent_10d" in data:
                            st.dataframe(pd.DataFrame(data["recent_10d"]), width="stretch")
                        elif isinstance(data, dict) and "recent" in data:
                            st.dataframe(pd.DataFrame(data["recent"]), width="stretch")
                        elif isinstance(data, dict) and "upcoming" in data:
                            st.dataframe(pd.DataFrame(data["upcoming"]), width="stretch")
                        elif isinstance(data, dict):
                            st.json(data)
                        else:
                            st.text(str(data))
            except Exception as e:
                st.error(f"研究失败：{repr(e)[:300]}")

# -------- 下载管理（v0.23：A/港/美股下载 + 缓存清理 + 进度展示） --------
elif page == "下载管理":
    import subprocess as _sp
    import pathlib as _pl
    from eq.db import DEFAULT_HOME as _HOME

    st.header("下载管理")
    st.caption("GUI 替代命令行管理数据下载，含缓存清理")

    _dl_tab1, _dl_tab2, _dl_tab3 = st.tabs(["A股下载", "港股下载", "美股下载"])

    # --- A股下载 ---
    with _dl_tab1:
        st.subheader("A股日线（腾讯 API → qlib .bin）")
        col_a1, col_a2, col_a3, col_a4 = st.columns(4)
        universe = col_a1.selectbox("Universe", ["csi300", "csi500", "csi800", "all", "watchlist"], index=1, key="a_universe")
        start_a = col_a2.text_input("起始日", "2024-01-01", key="a_start")
        end_a = col_a3.text_input("结束日", dt.date.today().isoformat(), key="a_end")
        workers_a = col_a4.number_input("并发", min_value=1, max_value=32, value=8, step=1, key="a_workers")
        extra_a = st.text_input("附加股票（逗号分隔，如 SH688256,SZ000001）", "", key="a_extra")
        col_a5, col_a6 = st.columns(2)
        extra_codes = [x.strip() for x in extra_a.split(",") if x.strip()] if extra_a else None
        if col_a5.button("📥 开始下载", type="primary", key="a_btn_dl"):
            cmd = ["eq", "ml", "update-data", "-u", universe, "-s", start_a, "-e", end_a, "-w", str(workers_a)]
            if extra_codes:
                cmd += ["-x", ",".join(extra_codes)]
            st.info(f"执行：{' '.join(cmd)}")
            with st.spinner("下载中... 完成后页面自动刷新"):
                proc = _sp.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if proc.returncode == 0:
                st.success("✅ 下载完成")
                st.code(proc.stdout[-2000:])
            else:
                st.error("❌ 下载失败")
                st.code(proc.stderr[-2000:] or proc.stdout[-2000:])
        if col_a6.button("🔄 重建 instruments", key="a_btn_regen"):
            cmd = ["eq", "ml", "regen-instruments", universe]
            if extra_codes:
                cmd += ["-x", ",".join(extra_codes)]
            proc = _sp.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            st.success("✅ instruments 重建完成" if proc.returncode == 0 else "❌ 失败")
            st.code(proc.stdout[-1000:] or proc.stderr[-1000:])

    # --- 港股下载 ---
    with _dl_tab2:
        st.subheader("港股日线（东财 push2his 主源，akshare 新浪源 fallback）")
        col_h1, col_h2, col_h3 = st.columns(3)
        top_h = col_h1.number_input("前 N 只", min_value=1, max_value=500, value=100, step=10, key="h_top")
        start_h = col_h2.text_input("起始日", "2024-01-01", key="h_start")
        codes_file_h = col_h3.text_input("品种表 txt（可选，留空用 top）", "", key="h_codes")
        if st.button("📥 开始下载港股", type="primary", key="h_btn_dl"):
            cmd = ["eq", "data", "hk", "-n", str(top_h), "-s", start_h]
            if codes_file_h.strip():
                # `eq data hk` 只有 --codes（逗号分隔代码），没有 --codes-file，
                # 此前直接传 --codes-file 会被 typer 拒掉、整个下载失败。
                # 这里先在本地把品种表解析成代码列表再传。
                try:
                    from eq.data.hk_market import parse_hk_codes_from_file
                    _codes = parse_hk_codes_from_file(codes_file_h.strip(), verbose=False)
                except Exception as _e:
                    st.error(f"品种表解析失败：{repr(_e)[:200]}")
                    _codes = []
                if _codes:
                    cmd += ["--codes", ",".join(_codes)]
                    st.caption(f"品种表解析出 {len(_codes)} 只港股")
                else:
                    st.warning("品种表未解析出港股代码，回退用「前 N 只」")
            st.info(f"执行：{' '.join(cmd)[:300]}")
            with st.spinner("下载中（东财源约 30 秒/100 只）..."):
                proc = _sp.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if proc.returncode == 0:
                st.success("✅ 港股下载完成")
                st.code(proc.stdout[-2000:])
            else:
                st.error("❌ 港股下载失败")
                st.code(proc.stderr[-2000:] or proc.stdout[-2000:])

        # 港股自选单股下载
        st.markdown("---")
        st.subheader("港股自选单股下载（东财源，秒级）")
        col_hs1, col_hs2, col_hs3 = st.columns([2, 2, 1])
        single_h = col_hs1.text_input("港股代码（5 位数字，如 00700）", "", key="h_single")
        single_start_h = col_hs2.text_input("起始日", "2024-01-01", key="h_single_start")
        if col_hs3.button("📥 下载单股", type="primary", key="h_btn_single"):
            if not single_h.strip():
                st.error("请填港股代码")
            else:
                cmd = ["eq", "data", "hk", "-n", "1", "-s", single_start_h, "--codes", single_h.strip().zfill(5)]
                st.info(f"执行：{' '.join(cmd)}")
                proc = _sp.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
                if proc.returncode == 0:
                    st.success("✅ 单股下载完成")
                    st.code(proc.stdout[-1500:])
                else:
                    st.error("❌ 失败")
                    st.code(proc.stderr[-1500:] or proc.stdout[-1500:])

        # 港股分钟线
        st.markdown("---")
        st.subheader("港股分钟线（东财 push2his 主源，yfinance fallback）")
        col_hm1, col_hm2, col_hm3 = st.columns(3)
        freq_hm = col_hm1.selectbox("频率", ["5min", "1min"], index=0, key="hk_min_freq")
        top_hm = col_hm2.number_input("前 N 只", min_value=1, max_value=500, value=100, step=10, key="hk_min_top")
        codes_file_hm = col_hm3.text_input("品种表 txt（可选）", "", key="hk_min_codes")
        if st.button("📥 下载分钟线港股", key="h_btn_min"):
            cmd = ["eq", "data", f"hk-{freq_hm}", "-n", str(top_hm)]
            if codes_file_hm.strip():
                cmd += ["--codes-file", codes_file_hm.strip()]
            st.info(f"执行：{' '.join(cmd)}（东财主源，无限流；东财失败才走 yfinance fallback）")
            with st.spinner("下载中（东财源约 30 秒/100 只）..."):
                proc = _sp.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if proc.returncode == 0:
                st.success("✅ 港股分钟线下载完成")
                st.code(proc.stdout[-2000:])
            else:
                st.error("❌ 失败")
                st.code(proc.stderr[-2000:] or proc.stdout[-2000:])

    # --- 美股下载 ---
    with _dl_tab3:
        st.subheader("美股日线（东财 push2his 主源，yfinance fallback）")
        col_u1, col_u2, col_u3 = st.columns(3)
        top_u = col_u1.number_input("前 N 只", min_value=1, max_value=500, value=100, step=10, key="us_top")
        start_u = col_u2.text_input("起始日", "2024-01-01", key="us_start")
        codes_u = col_u3.text_input("指定代码（可选，逗号分隔如 AAPL,MSFT）", "", key="us_codes")
        if st.button("📥 开始下载美股", type="primary", key="us_btn_dl"):
            cmd = ["eq", "data", "us", "-n", str(top_u), "-s", start_u]
            if codes_u.strip():
                # `eq data us` 接受的是 --codes（逗号分隔），此前这里传的是
                # 不存在的 --codes-file，任何填了内容的下载都必然失败。
                cmd += ["--codes", ",".join(c.strip().upper() for c in codes_u.split(",") if c.strip())]
            st.info(f"执行：{' '.join(cmd)}")
            with st.spinner("下载中（东财源约 30 秒/100 只）..."):
                proc = _sp.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if proc.returncode == 0:
                st.success("✅ 美股下载完成")
                st.code(proc.stdout[-2000:])
            else:
                st.error("❌ 失败")
                st.code(proc.stderr[-2000:] or proc.stdout[-2000:])

        # 美股自选单股下载
        st.markdown("---")
        st.subheader("美股自选单股下载（东财源，秒级）")
        col_us1, col_us2, col_us3 = st.columns([2, 2, 1])
        single_u = col_us1.text_input("美股代码（如 AAPL, MSFT）", "", key="us_single")
        single_start_u = col_us2.text_input("起始日", "2024-01-01", key="us_single_start")
        if col_us3.button("📥 下载单股", type="primary", key="us_btn_single"):
            if not single_u.strip():
                st.error("请填美股代码")
            else:
                cmd = ["eq", "data", "us", "-n", "1", "-s", single_start_u, "--codes", single_u.strip().upper()]
                st.info(f"执行：{' '.join(cmd)}")
                proc = _sp.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
                if proc.returncode == 0:
                    st.success("✅ 单股下载完成")
                    st.code(proc.stdout[-1500:])
                else:
                    st.error("❌ 失败")
                    st.code(proc.stderr[-1500:] or proc.stdout[-1500:])

    # --- 缓存清理（跨 tab 共用） ---
    st.markdown("---")
    st.subheader("🧹 缓存清理")
    # 此前这里写死了相对路径 Path("data")，结果取决于 streamlit 的工作目录——
    # 从项目根启动时指向不存在的 ./data，"清理"和"占用统计"全是空的。
    # 改用 eq.data.paths 里的绝对路径常量。
    from eq.data.paths import (
        DATA_ROOT as _DATA_ROOT, HK_1M_DIR as _HK1M, HK_5M_DIR as _HK5M,
        HK_DAILY_DIR as _HKD, QLIB_CN_DATA_DIR as _QLIB, US_DAILY_DIR as _USD,
    )

    _qlib_root = _pl.Path(_QLIB)
    _cache_dirs = {
        "A 股 qlib .bin（features）": _qlib_root / "features",
        "A 股 qlib 日历": _qlib_root / "calendars",
        "A 股 qlib instruments": _qlib_root / "instruments",
        "港股日线 CSV": _pl.Path(_HKD),
        "港股 5 分钟 CSV": _pl.Path(_HK5M),
        "港股 1 分钟 CSV": _pl.Path(_HK1M),
        "港股特征 CSV": _pl.Path(_DATA_ROOT) / "hk" / "features",
        "美股日线 CSV": _pl.Path(_USD),
        "回测结果 parquet": _pl.Path(_HOME) / "backtests",
        "导出文件": _pl.Path(_HOME) / "exports",
    }
    cache_choice = st.multiselect("选择要清理的缓存目录", list(_cache_dirs.keys()))
    col_c1, col_c2 = st.columns(2)
    if col_c1.button("🧹 清理选中缓存", type="primary", key="cache_btn_clean"):
        cleared = 0
        for name in cache_choice:
            d = _cache_dirs[name]
            if d.exists():
                try:
                    for f in d.rglob("*"):
                        if f.is_file():
                            f.unlink()
                    cleared += 1
                    st.info(f"已清 {name}")
                except Exception as e:
                    st.error(f"清 {name} 失败：{repr(e)[:100]}")
        st.success(f"清理完成，共清 {cleared} 个目录" if cleared else "未选任何目录")
    if col_c2.button("📊 查看缓存占用", key="cache_btn_view"):
        st.session_state["cache_viewed"] = True
    # rerun 时若已查看过，仍渲染 DataFrame（按钮 reset 后不丢）
    if st.session_state.get("cache_viewed"):
        rows = []
        for name, d in _cache_dirs.items():
            if d.exists():
                total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                files = sum(1 for f in d.rglob("*") if f.is_file())
                rows.append({"缓存": name, "文件数": files, "大小 MB": round(total / 1024 / 1024, 2)})
            else:
                rows.append({"缓存": name, "文件数": 0, "大小 MB": 0.0})
        st.dataframe(pd.DataFrame(rows), width="stretch")

st.sidebar.divider()


def _version() -> str:
    """版本号。写死的话每次发版都会忘记改（挂了很久的 "v0.23" 就是这么来的）。

    **优先读 pyproject.toml**：源码开发时 ``pip install -e`` 记录的
    installed metadata 会停在安装那一刻的版本（实测显示 v0.26 而仓库已到
    v0.32）。仓库里有 pyproject 就以它为准；pip 安装的场景没有 pyproject，
    再退回 installed metadata。
    """
    try:
        import tomllib
        from pathlib import Path

        pt = Path(__file__).resolve().parents[2] / "pyproject.toml"
        if pt.exists():
            return tomllib.loads(pt.read_text(encoding="utf-8"))["project"]["version"]
    except Exception:
        pass
    try:
        from importlib.metadata import version

        return version("eternityquant")
    except Exception:
        return "dev"


_foot = f"EternityQuant v{_version()} · Streamlit 仪表盘"
if _theme_info.get("enabled"):
    _foot += f" · 主题 {_theme_info['cfg'].image.name}"
st.sidebar.caption(_foot)
