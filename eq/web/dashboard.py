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
    ["概览", "持仓", "自选", "监控规则", "ML 模型", "深度研究"],
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

st.sidebar.divider()
st.sidebar.caption("EternityQuant v0.11 · Streamlit 仪表盘")
