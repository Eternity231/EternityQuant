"""Streamlit 仪表盘主入口（被 streamlit run 直接执行）。

侧边栏分页：
- 概览（持仓 + 自选 + 最新信号汇总）
- 持仓
- 自选
- 监控规则
- ML 模型
- 回测
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from eq.core import monitor as mon_svc
from eq.core import portfolio as pf_svc
from eq.core import watchlist as wl_svc
from eq.db import get_state_conn
from eq.strategy.factors import ml as ml_svc

st.set_page_config(page_title="EternityQuant", page_icon="📊", layout="wide")


def _fmt_df(rows: list[dict]) -> pd.DataFrame:
    """列表字典转 DataFrame 显示。"""
    return pd.DataFrame(rows) if rows else pd.DataFrame()


st.title("EternityQuant 个人散户量化助手")
st.caption(f"今日 {dt.date.today().isoformat()}")

page = st.sidebar.selectbox(
    "页面",
    ["概览", "持仓", "自选", "监控规则", "ML 模型"],
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

# -------- ML 模型 --------
elif page == "ML 模型":
    st.header("ML 模型")
    df = _fmt_df(ml_svc.list_models())
    if df.empty:
        st.info("无模型记录")
    else:
        st.dataframe(df, use_container_width=True)
        active = df[df["is_active"] == 1] if "is_active" in df.columns else pd.DataFrame()
        st.subheader("当前激活模型")
        if active.empty:
            st.warning("无激活模型")
        else:
            st.dataframe(active, use_container_width=True)

st.sidebar.divider()
st.sidebar.caption("EternityQuant v0.1 · Streamlit 仪表盘")
