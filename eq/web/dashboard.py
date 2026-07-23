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
import json

import pandas as pd
import streamlit as st

from eq.core import monitor as mon_svc
from eq.core import portfolio as pf_svc
from eq.core import watchlist as wl_svc
from eq.db import execute, get_state_conn
from eq.strategy.factors import ml as ml_svc

st.set_page_config(page_title="EternityQuant", page_icon="📊", layout="wide")


def _fmt_df(rows: list[dict]) -> pd.DataFrame:
    """列表字典转 DataFrame 显示。"""
    return pd.DataFrame(rows) if rows else pd.DataFrame()


st.title("EternityQuant 个人散户量化助手")
st.caption(f"今日 {dt.date.today().isoformat()}")

page = st.sidebar.selectbox(
    "页面",
    ["概览", "持仓", "自选", "监控规则", "ML 模型", "下载管理", "深度研究"],
    index=0,
)

# -------- 概览 --------
if page == "概览":
    st.header("概览")
    col1, col2, col3 = st.columns(3)
    col1.metric("自选股数", len(wl_svc.list_all()))
    col2.metric("持仓只数", len(pf_svc.list_open()))
    col3.metric("监控规则", len(mon_svc.list_rules()))
    st.subheader("当前持仓")
    df_pos = _fmt_df(pf_svc.list_open())
    if df_pos.empty:
        st.info("无持仓")
    else:
        st.dataframe(df_pos, use_container_width=True)
    st.subheader("最近监控触发")
    rules = mon_svc.list_rules()
    fired = [r for r in rules if r["fire_count"] > 0]
    if not fired:
        st.info("尚无规则触发记录")
    else:
        st.dataframe(_fmt_df(fired), use_container_width=True)

# -------- 持仓 --------
elif page == "持仓":
    st.header("当前持仓")
    df = _fmt_df(pf_svc.list_open())
    if df.empty:
        st.info("无持仓")
    else:
        st.dataframe(df, use_container_width=True)
        st.subheader("已实现盈亏合计")
        st.metric("未实现盈亏（按成本价）", f"{df['realized_pnl'].sum() if 'realized_pnl' in df else 0:+.2f}")
    st.subheader("已清仓记录")
    df_closed = _fmt_df(pf_svc.list_closed(limit=50))
    if df_closed.empty:
        st.info("无已清仓记录")
    else:
        st.dataframe(df_closed, use_container_width=True)

# -------- 自选 --------
elif page == "自选":
    st.header("自选股")
    df = _fmt_df(wl_svc.list_all())
    if df.empty:
        st.info("自选列表为空")
    else:
        st.dataframe(df, use_container_width=True)

# -------- 监控规则 --------
elif page == "监控规则":
    st.header("监控规则")
    rules = mon_svc.list_rules()
    if not rules:
        st.info("无监控规则")
    else:
        df = pd.DataFrame(rules)
        st.dataframe(df, use_container_width=True)
        st.subheader("触发统计")
        fired = [r for r in rules if r["fire_count"] > 0]
        st.metric("累计触发条数", sum(r["fire_count"] for r in rules))
        if fired:
            st.dataframe(_fmt_df(fired), use_container_width=True)

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
        st.dataframe(df, use_container_width=True)

        # 当前激活模型
        active = [m for m in models if m.get("is_active") == 1]
        st.subheader("当前激活模型")
        if not active:
            st.warning("无激活模型")
        else:
            st.dataframe(pd.DataFrame(active), use_container_width=True)

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
                            st.success(f"Top10 预测完成（已写入 ml_predictions 表）")
                            st.dataframe(pred_df, use_container_width=True)
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
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

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
                            st.dataframe(pd.DataFrame(data["recent_5d"]), use_container_width=True)
                        elif isinstance(data, dict) and "recent_10d" in data:
                            st.dataframe(pd.DataFrame(data["recent_10d"]), use_container_width=True)
                        elif isinstance(data, dict) and "recent" in data:
                            st.dataframe(pd.DataFrame(data["recent"]), use_container_width=True)
                        elif isinstance(data, dict) and "upcoming" in data:
                            st.dataframe(pd.DataFrame(data["upcoming"]), use_container_width=True)
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
                cmd += ["--codes-file", codes_file_h.strip()]
            st.info(f"执行：{' '.join(cmd)}")
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
        codes_file_u = col_u3.text_input("品种表 txt（可选）", "", key="us_codes")
        if st.button("📥 开始下载美股", type="primary", key="us_btn_dl"):
            cmd = ["eq", "data", "us", "-n", str(top_u), "-s", start_u]
            if codes_file_u.strip():
                cmd += ["--codes-file", codes_file_u.strip()]
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
    _cache_root = _pl.Path("data")
    _qlib_root = _pl.Path(_HOME) / "data" / "a" / "qlib_cn_data"
    _cache_dirs = {
        "A 股 qlib .bin（features）": _qlib_root / "features",
        "A 股 qlib 日历": _qlib_root / "calendars",
        "A 股 qlib instruments": _qlib_root / "instruments",
        "港股日线 CSV": _cache_root / "hk" / "daily",
        "港股 5 分钟 CSV": _cache_root / "hk" / "5m",
        "港股 1 分钟 CSV": _cache_root / "hk" / "1m",
        "美股日线 CSV": _cache_root / "us" / "daily",
        "美股 5 分钟 CSV": _cache_root / "us" / "5m",
        "美股 1 分钟 CSV": _cache_root / "us" / "1m",
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
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.sidebar.divider()
st.sidebar.caption("EternityQuant v0.23 · Streamlit 仪表盘")
