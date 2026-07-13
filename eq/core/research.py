"""个股深度研究引擎（v0.10）：调 vibe-trading MCP get_* 系列作数据补全。

vibe-trading MCP 是软依赖——框架自写全部核心引擎，但 research 时借其 80+ skill
做数据补全（基本面/资金流/新闻/研报/龙虎榜/解禁等），避免自己重复造轮子。

MCP 调用走 stdio（.mcp.json 配置），通过 mcp__vibe-trading__get_* 系列工具。
但 Python 进程内不能直接调 MCP 工具——MCP 工具是给 AI agent 用的，
所以这里走两条路：
1. AI agent 路径：用户在 AtomCode/Claude 等支持 MCP 的 agent 里跑 `eq research`
   时，agent 会看到 research 输出的"建议补全"提示，主动调 MCP get_* 工具补全
2. 直 API 路径：对核心数据（行情/基本面/新闻），直接用 akshare/yfinance/baostock
   走自写逻辑，不绕 MCP（更快更稳）

本文走路径 2 ——直 API 拉数据，不绕 MCP。MCP 工具留作 agent 交互式补全。
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd

from eq.data.market import detect_market, get_recent_bars, get_snapshot


def research(symbol: str, sections: list[str] | None = None) -> dict[str, Any]:
    """对个股做深度研究，按市场自动选数据源汇总。

    Args:
        symbol: 股票符号，如 600519.SH / AAPL.US / 00700.HK / BTC-USDT
        sections: 指定要拉的板块，缺省按市场全拉
            A股: snapshot/financial/fund_flow/news/research/block_trades/margin/shareholders/lockup/northbound/sector
            港股: snapshot/profile/news/fund_flow
            美股: snapshot/profile/sec_filings/news/financial/options
            加密: snapshot
    Returns:
        {"symbol": str, "market": str, "snapshot": dict, "financial": dict, ...}
    """
    market = detect_market(symbol)
    if sections is None:
        sections = _DEFAULT_SECTIONS.get(market, ["snapshot"])

    result: dict[str, Any] = {"symbol": symbol, "market": market, "as_of": dt.date.today().isoformat()}

    for sec in sections:
        handler = _SECTION_HANDLERS.get(sec)
        if handler is None:
            result[sec] = {"error": f"未知板块 {sec}"}
            continue
        try:
            result[sec] = handler(symbol, market)
        except Exception as e:
            result[sec] = {"error": f"{sec} 拉取失败：{repr(e)[:200]}"}

    return result


# ---------- 默认板块（按市场） ----------

_DEFAULT_SECTIONS = {
    "A": ["snapshot", "financial", "fund_flow", "news", "research", "block_trades",
          "margin", "shareholders", "lockup", "northbound", "sector"],
    "HK": ["snapshot", "profile", "news", "fund_flow"],
    "US": ["snapshot", "profile", "sec_filings", "news", "financial", "options"],
    "CRYPTO": ["snapshot"],
}


# ---------- 板块处理器 ----------

def _h_snapshot(symbol: str, market: str) -> dict[str, Any]:
    """行情快照 + 最近 30 日 K 线摘要。

    A 股走 baostock（稳），港/美走 yfinance（中国大陆限流）+ akshare fallback（东财限流），
    失败时返回 MCP 补全建议（agent 里跑会自动调 vibe-trading get_market_data 补全）。
    """
    try:
        snap = get_snapshot(symbol)
    except Exception as e:
        return {"hint": f"港/美行情拉取失败（{repr(e)[:100]}），建议调 vibe-trading MCP get_market_data([\"{symbol}\"], \"{dt.date.today().isoformat()}\", \"{dt.date.today().isoformat()}\")"}
    bars = get_recent_bars(symbol, days=30)
    # 30 日摘要：高/低/均价/量均
    if bars.empty:
        return {"snapshot": snap, "recent_30d": None}
    recent = {
        "high": float(bars["high"].max()),
        "low": float(bars["low"].min()),
        "avg_close": float(bars["close"].mean()),
        "avg_volume": float(bars["volume"].mean()),
        "total_amount": float(bars["amount"].sum()) if "amount" in bars.columns else None,
        "days": len(bars),
    }
    return {"snapshot": snap, "recent_30d": recent}


def _h_financial(symbol: str, market: str) -> dict[str, Any]:
    """基本面：A股财报指标（akshare 东财），港美走 vibe-trading MCP 补全建议。"""
    if market != "A":
        return {"hint": f"港/美股基本面建议调 vibe-trading MCP get_financial_statements({symbol})"}
    import akshare as ak
    # A股代码剥离 .SH/.SZ/.BJ 后缀
    bare = symbol.split(".")[0]
    try:
        df = ak.stock_individual_info_em(symbol=bare)
        # df 是两列 DataFrame：item / value
        info = dict(zip(df.iloc[:, 0], df.iloc[:, 1].astype(str)))
        return {"info": info}
    except Exception as e:
        return {"error": f"akshare stock_individual_info_em 失败：{repr(e)[:150]}"}


def _h_fund_flow(symbol: str, market: str) -> dict[str, Any]:
    """资金流向：A股东财资金流向，港美走 MCP 补全建议。"""
    if market != "A":
        return {"hint": f"港/美资金流建议调 vibe-trading MCP get_fund_flow([\"{symbol}\"])"}
    import akshare as ak
    bare = symbol.split(".")[0]
    # 个股资金流向（东财）
    for fn_name in ["stock_individual_fund_flow", "stock_individual_fund_flow_rank"]:
        try:
            fn = getattr(ak, fn_name)
            df = fn(stock=bare, market="sh" if symbol.endswith(".SH") else "sz")
            if not df.empty:
                # 取最近 5 日
                recent = df.head(5) if hasattr(df, "head") else df[:5]
                return {"recent_5d": recent.to_dict("records") if hasattr(recent, "to_dict") else recent}
        except Exception:
            continue
    return {"error": "akshare 资金流向接口失败"}


def _h_news(symbol: str, market: str) -> dict[str, Any]:
    """新闻：A股 akshare 东财，港美走 MCP 补全建议。"""
    if market == "A":
        import akshare as ak
        bare = symbol.split(".")[0]
        try:
            df = ak.stock_news_em(symbol=bare)
            return {"headlines": df.head(10).to_dict("records") if hasattr(df, "to_dict") else df[:10]}
        except Exception as e:
            return {"error": f"akshare stock_news_em 失败：{repr(e)[:150]}"}
    return {"hint": f"港/美新闻建议调 vibe-trading MCP get_stock_news(\"{symbol}\")"}


def _h_research(symbol: str, market: str) -> dict[str, Any]:
    """研报：A股 akshare 东财，港美走 MCP 补全建议。"""
    if market != "A":
        return {"hint": f"港/美研报建议调 vibe-trading MCP get_research_reports(\"{symbol}\")"}
    import akshare as ak
    bare = symbol.split(".")[0]
    for fn_name in ["stock_research_report_em", "stock_notice_report"]:
        try:
            fn = getattr(ak, fn_name)
            df = fn(symbol=bare) if fn_name == "stock_research_report_em" else fn(symbol=bare)
            if not df.empty:
                return {"reports": df.head(10).to_dict("records") if hasattr(df, "to_dict") else df[:10]}
        except Exception:
            continue
    return {"hint": "akshare 研报接口不可用，建议调 vibe-trading MCP get_research_reports"}


def _h_block_trades(symbol: str, market: str) -> dict[str, Any]:
    """大宗交易：A股东财，港美无。"""
    if market != "A":
        return {"hint": "大宗交易仅 A 股有"}
    import akshare as ak
    bare = symbol.split(".")[0]
    try:
        df = ak.stock_dzjy_sctj()  # 大宗交易市场统计，全市场
        # 过滤本股
        if not df.empty and "代码" in df.columns:
            mine = df[df["代码"].astype(str).str.contains(bare)]
            if not mine.empty:
                return {"recent": mine.head(10).to_dict("records")}
        return {"hint": "近端无大宗交易"}
    except Exception as e:
        return {"error": f"akshare 大宗交易失败：{repr(e)[:150]}"}


def _h_margin(symbol: str, market: str) -> dict[str, Any]:
    """融资融券：A股东财，港美无。"""
    if market != "A":
        return {"hint": "融资融券仅 A 股有"}
    import akshare as ak
    bare = symbol.split(".")[0]
    try:
        df = ak.stock_margin_detail_szse() if symbol.endswith(".SZ") else ak.stock_margin_detail_sse()
        # 过滤本股
        if not df.empty:
            code_col = "证券代码" if "证券代码" in df.columns else "代码"
            mine = df[df[code_col].astype(str).str.contains(bare)]
            if not mine.empty:
                return {"recent": mine.head(5).to_dict("records")}
        return {"hint": "近端无融资融券明细"}
    except Exception as e:
        return {"error": f"akshare 融资融券失败：{repr(e)[:150]}"}


def _h_shareholders(symbol: str, market: str) -> dict[str, Any]:
    """股东户数：A股东财，港美无。"""
    if market != "A":
        return {"hint": "股东户数仅 A 股有"}
    import akshare as ak
    bare = symbol.split(".")[0]
    try:
        df = ak.stock_zh_a_gdhs(symbol=bare)
        return {"recent": df.head(5).to_dict("records") if hasattr(df, "to_dict") else df[:5]}
    except Exception as e:
        return {"error": f"akshare 股东户数失败：{repr(e)[:150]}"}


def _h_lockup(symbol: str, market: str) -> dict[str, Any]:
    """解禁：A股东财，港美无。"""
    if market != "A":
        return {"hint": "解禁仅 A 股有"}
    import akshare as ak
    bare = symbol.split(".")[0]
    try:
        df = ak.stock_restricted_release_summary_sina()  # 全市场解禁
        if not df.empty and "代码" in df.columns:
            mine = df[df["代码"].astype(str).str.contains(bare)]
            if not mine.empty:
                return {"upcoming": mine.head(5).to_dict("records")}
        return {"hint": "近端无解禁"}
    except Exception as e:
        return {"error": f"akshare 解禁失败：{repr(e)[:150]}"}


def _h_northbound(symbol: str, market: str) -> dict[str, Any]:
    """北向资金：A股东财，港美无。"""
    if market != "A":
        return {"hint": "北向资金仅 A 股相关"}
    import akshare as ak
    try:
        df = ak.stock_hsgt_hist_em(symbol="北向资金")
        if not df.empty:
            recent = df.tail(10)
            return {"recent_10d": recent.to_dict("records")}
        return {"hint": "北向资金数据为空"}
    except Exception as e:
        return {"error": f"akshare 北向资金失败：{repr(e)[:150]}"}


def _h_sector(symbol: str, market: str) -> dict[str, Any]:
    """板块归属：A股东财，港美无。"""
    if market != "A":
        return {"hint": "板块归属仅 A 股有"}
    import akshare as ak
    bare = symbol.split(".")[0]
    try:
        df = ak.stock_board_industry_name_em()  # 行业板块列表
        # 找本股所属行业（需要逐板块查 constituent，简化第一版只返回板块列表）
        return {"industries_total": len(df), "hint": "个股所属板块明细建议调 vibe-trading MCP get_sector_info"}
    except Exception as e:
        return {"error": f"akshare 板块失败：{repr(e)[:150]}"}


def _h_profile(symbol: str, market: str) -> dict[str, Any]:
    """公司画像：港美走 MCP 补全建议。"""
    return {"hint": f"港/美公司画像建议调 vibe-trading MCP get_stock_profile(\"{symbol}\")"}


def _h_sec_filings(symbol: str, market: str) -> dict[str, Any]:
    """SEC 公告：美股走 MCP 补全建议。"""
    if market != "US":
        return {"hint": "SEC 公告仅美股有"}
    return {"hint": f"SEC 公告建议调 vibe-trading MCP get_sec_filings(\"{symbol.split('.')[0]}\")"}


def _h_options(symbol: str, market: str) -> dict[str, Any]:
    """期权链：美股走 MCP 补全建议。"""
    if market != "US":
        return {"hint": "期权链仅美股有"}
    return {"hint": f"期权链建议调 vibe-trading MCP get_options_chain(\"{symbol.split('.')[0]}\")"}


# ---------- 注册 ----------

_SECTION_HANDLERS = {
    "snapshot": _h_snapshot,
    "financial": _h_financial,
    "fund_flow": _h_fund_flow,
    "news": _h_news,
    "research": _h_research,
    "block_trades": _h_block_trades,
    "margin": _h_margin,
    "shareholders": _h_shareholders,
    "lockup": _h_lockup,
    "northbound": _h_northbound,
    "sector": _h_sector,
    "profile": _h_profile,
    "sec_filings": _h_sec_filings,
    "options": _h_options,
}


# ---------- 格式化输出 ----------

def format_research(report: dict[str, Any]) -> str:
    """格式化深度研究报告为文本。"""
    sym = report["symbol"]
    market = report["market"]
    market_label = {"A": "A 股", "HK": "港股", "US": "美股", "CRYPTO": "加密"}.get(market, market)

    lines = [f"\n{'=' * 60}", f"  {sym} 深度研究报告  {market_label}  {report.get('as_of', '')}", f"{'=' * 60}\n"]

    for sec, data in report.items():
        if sec in ("symbol", "market", "as_of"):
            continue
        sec_label = _SECTION_LABELS.get(sec, sec)
        lines.append(f"--- {sec_label} ---")
        if isinstance(data, dict) and "error" in data:
            lines.append(f"  ❌ {data['error']}")
        elif isinstance(data, dict) and "hint" in data:
            lines.append(f"  💡 {data['hint']}")
        elif isinstance(data, dict) and "snapshot" in data:
            # snapshot 板块特殊
            snap = data["snapshot"]
            lines.append(f"  最新价 {snap.get('close', '?')}  涨跌幅 {snap.get('change_pct', '?'):+}%")
            lines.append(f"  今开 {snap.get('open', '?')}  最高 {snap.get('high', '?')}  最低 {snap.get('low', '?')}")
            lines.append(f"  成交量 {snap.get('volume', '?')}  成交额 {snap.get('amount', '?')}")
            if data.get("recent_30d"):
                r = data["recent_30d"]
                lines.append(f"  近 {r['days']} 日：高 {r['high']:.2f}  低 {r['low']:.2f}  均价 {r['avg_close']:.2f}")
        elif isinstance(data, dict) and "info" in data:
            info = data["info"]
            for k, v in list(info.items())[:15]:
                lines.append(f"  {k}: {v}")
        elif isinstance(data, dict) and "recent_5d" in data:
            for row in data["recent_5d"][:5]:
                lines.append(f"  {row}")
        elif isinstance(data, dict) and "headlines" in data:
            for h in data["headlines"][:5]:
                if isinstance(h, dict):
                    title = h.get("新闻标题") or h.get("title") or str(h)[:60]
                    lines.append(f"  • {title}")
        elif isinstance(data, dict) and "reports" in data:
            for r in data["reports"][:5]:
                if isinstance(r, dict):
                    title = r.get("研报标题") or r.get("title") or str(r)[:60]
                    lines.append(f"  • {title}")
        elif isinstance(data, dict) and "recent" in data:
            for r in data.get("recent", [])[:5]:
                lines.append(f"  {r}")
        elif isinstance(data, dict) and "recent_10d" in data:
            for r in data["recent_10d"][:5]:
                if isinstance(r, dict):
                    date = r.get("日期") or r.get("date") or ""
                    flow = r.get("当日成交净买额") or r.get("value") or ""
                    lines.append(f"  {date}: {flow}")
        elif isinstance(data, dict):
            for k, v in list(data.items())[:8]:
                lines.append(f"  {k}: {v}")
        else:
            lines.append(f"  {data}")
        lines.append("")

    return "\n".join(lines)


_SECTION_LABELS = {
    "snapshot": "行情快照",
    "financial": "基本面",
    "fund_flow": "资金流向",
    "news": "新闻",
    "research": "研报",
    "block_trades": "大宗交易",
    "margin": "融资融券",
    "shareholders": "股东户数",
    "lockup": "解禁",
    "northbound": "北向资金",
    "sector": "板块归属",
    "profile": "公司画像",
    "sec_filings": "SEC 公告",
    "options": "期权链",
}
