"""个股深度研究引擎（v0.10；v0.34 全部本地化）。

**不依赖任何外部服务或 AI 助手**——每个板块都由本进程直接调数据 SDK 取数：
A 股走 akshare（东财/新浪），港美走 yfinance，两者都是本项目的硬依赖。

历史包袱：v0.10~v0.33 里港美的 profile / financial / news / research /
sec_filings / options 六个板块**根本没有实现**，只返回一句"建议用某某外部工具
补全"的提示，指望用户恰好在能调那个工具的环境里跑 `eq research`。
结果是港股 4 个板块里 3 个、美股 6 个里 5 个都是占位符——命令行单独跑等于没内容。
v0.34 用 yfinance 全部落地，纯 Python 可跑。

拿不到数据时返回 ``{"note": ...}``，说明为什么没有、该看哪个板块替代，
而不是把责任推给别的工具。
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from eq.data.market import yfinance_symbol, detect_market, get_recent_bars, get_snapshot


def _yf_ticker(symbol: str, market: str):
    """项目符号 → yfinance Ticker。延迟导入：未装 yfinance 时不阻塞 CLI 启动。"""
    import yfinance as yf

    return yf.Ticker(yfinance_symbol(symbol, market))


def research(symbol: str, sections: list[str] | None = None) -> dict[str, Any]:
    """对个股做深度研究，按市场自动选数据源汇总。

    Args:
        symbol: 股票符号，如 600519.SH / AAPL.US / 00700.HK / BTC-USDT
        sections: 指定要拉的板块，缺省按市场全拉
            A股: snapshot/financial/fund_flow/news/research/block_trades/margin/shareholders/lockup/northbound/sector
            港股: snapshot/profile/financial/news/research/holders
            美股: snapshot/profile/financial/news/research/sec_filings/options/holders
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
    # v0.34：港美从「1 个真板块 + 一堆占位」补成全部可用。
    # 港股不放 fund_flow——没有免费的港股个股资金流接口，放个必然返回
    # "无数据"的板块只是浪费一次请求，用 holders（机构持股）替代更实在。
    "HK": ["snapshot", "profile", "financial", "news", "research", "holders"],
    "US": ["snapshot", "profile", "financial", "news", "research",
           "sec_filings", "options", "holders"],
    "CRYPTO": ["snapshot"],
}


# ---------- 板块处理器 ----------

def _h_snapshot(symbol: str, market: str) -> dict[str, Any]:
    """行情快照 + 最近 30 日 K 线摘要。

    A 股走 baostock（稳），港/美走 yfinance + akshare fallback。
    """
    try:
        snap = get_snapshot(symbol)
    except Exception as e:
        return {"note": f"行情拉取失败（{repr(e)[:100]}）。"
                        "跑 `eq doctor` 查各数据源连通性，或换个源重试。"}
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


# yfinance info 里挑得出来的估值/盈利指标：(键, 中文名, 单位)
#   raw  = 原样显示     frac = 小数比例，×100 显示     pct = **本来就是百分数**
#
# 最后一档是 yfinance 的一个坑：同一个 info 字典里，profitMargins/ROE/增速
# 都是小数（0.2715 = 27.15%），唯独 dividendYield 早已改成百分数
# （AAPL 返回 0.32，实际股息率 0.32%，不是 32%）。一律 ×100 会把股息率放大 100 倍。
# 校验方式：dividendRate / currentPrice —— 1.08 / 333.02 = 0.324%，对得上 0.32。
_VALUATION_KEYS = [
    ("marketCap", "市值", "raw"),
    ("trailingPE", "市盈率TTM", "raw"),
    ("forwardPE", "预期市盈率", "raw"),
    ("priceToBook", "市净率", "raw"),
    ("dividendYield", "股息率", "pct"),
    ("profitMargins", "净利率", "frac"),
    ("returnOnEquity", "ROE", "frac"),
    ("debtToEquity", "负债权益比", "raw"),
    ("revenueGrowth", "营收增速", "frac"),
    ("earningsGrowth", "利润增速", "frac"),
]


def _h_financial(symbol: str, market: str) -> dict[str, Any]:
    """基本面。A 股走 akshare 东财；港美走 yfinance 财报 + 估值指标（v0.34 落地）。"""
    if market != "A":
        t = _yf_ticker(symbol, market)
        out: dict[str, Any] = {}
        try:
            info = t.info or {}
            metrics = {}
            for key, label, unit in _VALUATION_KEYS:
                v = info.get(key)
                if v is None:
                    continue
                if unit == "frac":
                    metrics[label] = f"{v * 100:.2f}%"
                elif unit == "pct":
                    metrics[label] = f"{v:.2f}%"
                else:
                    metrics[label] = v
            if metrics:
                out["metrics"] = metrics
        except Exception as e:
            out["metrics_error"] = repr(e)[:120]
        # 利润表最近 3 期的关键行（yfinance 返回列=报告期、行=科目）
        try:
            stmt = t.income_stmt
            if stmt is not None and not stmt.empty:
                rows = [r for r in ("Total Revenue", "Gross Profit", "Operating Income",
                                    "Net Income") if r in stmt.index]
                cols = list(stmt.columns)[:3]
                # 列名是 Timestamp，str()[:10] 就是报告期日期
                out["income_stmt"] = {
                    str(c)[:10]: {
                        # NaN != NaN，用它判空，避开对 numpy/pandas 缺失值类型的依赖
                        r: (None if stmt.loc[r, c] != stmt.loc[r, c] else float(stmt.loc[r, c]))
                        for r in rows
                    }
                    for c in cols
                }
        except Exception as e:
            out.setdefault("stmt_error", repr(e)[:120])
        return out or {"note": f"yfinance 未返回 {symbol} 的基本面数据（可能是标的太冷门或临时限流）"}
    import akshare as ak
    # A股代码剥离 .SH/.SZ/.BJ 后缀
    bare = symbol.split(".")[0]
    try:
        df = ak.stock_individual_info_em(symbol=bare)
        # df 是两列 DataFrame：item / value
        info = dict(zip(df.iloc[:, 0], df.iloc[:, 1].astype(str), strict=False))
        return {"info": info}
    except Exception as e:
        return {"error": f"akshare stock_individual_info_em 失败：{repr(e)[:150]}"}


def _h_fund_flow(symbol: str, market: str) -> dict[str, Any]:
    """资金流向：A 股东财逐日主力净流入。

    港美**没有**免费的个股资金流接口（交易所不公开逐笔主动买卖方向，
    东财那套"主力/超大单"是 A 股独有的分类）。与其编一个假指标，不如说清楚
    ——想看港美的钱往哪走，用 holders 板块的机构持股变化。
    """
    if market != "A":
        return {"note": "港美无公开的个股资金流数据（A 股独有）。港美看 holders 板块的机构持股。"}
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
    """新闻：A 股 akshare 东财，港美 yfinance（v0.34 落地）。"""
    if market != "A":
        try:
            items = _yf_ticker(symbol, market).news or []
        except Exception as e:
            return {"note": f"yfinance 新闻拉取失败：{repr(e)[:120]}"}
        heads = []
        for n in items[:10]:
            # yfinance 1.x 把字段挪进了 content 子字典，两种结构都兼容
            c = n.get("content", n) if isinstance(n, dict) else {}
            title = c.get("title") or n.get("title")
            if not title:
                continue
            pub = c.get("pubDate") or c.get("providerPublishTime") or ""
            src = (c.get("provider") or {}).get("displayName") if isinstance(
                c.get("provider"), dict) else n.get("publisher", "")
            heads.append({"title": title, "publisher": src, "time": str(pub)[:19]})
        return {"headlines": heads} if heads else {"note": "yfinance 未返回该标的的新闻"}
    if market == "A":
        import akshare as ak
        bare = symbol.split(".")[0]
        try:
            df = ak.stock_news_em(symbol=bare)
            return {"headlines": df.head(10).to_dict("records") if hasattr(df, "to_dict") else df[:10]}
        except Exception as e:
            return {"error": f"akshare stock_news_em 失败：{repr(e)[:150]}"}
    return {"note": "未取到新闻"}


def _h_research(symbol: str, market: str) -> dict[str, Any]:
    """研报/分析师观点。

    A 股是券商研报标题（akshare 东财）；港美 yfinance 给不出研报原文，
    但给得出**分析师目标价和评级分布**——对散户其实更直接（v0.34 落地）。
    """
    if market != "A":
        t = _yf_ticker(symbol, market)
        out: dict[str, Any] = {}
        try:
            tgt = t.analyst_price_targets
            if isinstance(tgt, dict) and tgt:
                out["targets"] = {k: v for k, v in tgt.items() if v is not None}
        except Exception:
            pass
        try:
            rec = t.recommendations
            if rec is not None and not rec.empty:
                out["ratings"] = rec.head(4).to_dict("records")
        except Exception:
            pass
        return out or {"note": "yfinance 未返回分析师数据"}
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
    return {"note": "akshare 研报接口暂不可用"}


def _h_block_trades(symbol: str, market: str) -> dict[str, Any]:
    """大宗交易：A股东财，港美无。"""
    if market != "A":
        return {"note": "大宗交易仅 A 股有"}
    import akshare as ak
    bare = symbol.split(".")[0]
    try:
        df = ak.stock_dzjy_sctj()  # 大宗交易市场统计，全市场
        # 过滤本股
        if not df.empty and "代码" in df.columns:
            mine = df[df["代码"].astype(str).str.contains(bare)]
            if not mine.empty:
                return {"recent": mine.head(10).to_dict("records")}
        return {"note": "近端无大宗交易"}
    except Exception as e:
        return {"error": f"akshare 大宗交易失败：{repr(e)[:150]}"}


def _h_margin(symbol: str, market: str) -> dict[str, Any]:
    """融资融券：A股东财，港美无。"""
    if market != "A":
        return {"note": "融资融券仅 A 股有"}
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
        return {"note": "近端无融资融券明细"}
    except Exception as e:
        return {"error": f"akshare 融资融券失败：{repr(e)[:150]}"}


def _h_shareholders(symbol: str, market: str) -> dict[str, Any]:
    """股东户数：A股东财，港美无。"""
    if market != "A":
        return {"note": "股东户数仅 A 股有"}
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
        return {"note": "解禁仅 A 股有"}
    import akshare as ak
    bare = symbol.split(".")[0]
    try:
        df = ak.stock_restricted_release_summary_sina()  # 全市场解禁
        if not df.empty and "代码" in df.columns:
            mine = df[df["代码"].astype(str).str.contains(bare)]
            if not mine.empty:
                return {"upcoming": mine.head(5).to_dict("records")}
        return {"note": "近端无解禁"}
    except Exception as e:
        return {"error": f"akshare 解禁失败：{repr(e)[:150]}"}


def _h_northbound(symbol: str, market: str) -> dict[str, Any]:
    """北向资金：A股东财，港美无。"""
    if market != "A":
        return {"note": "北向资金仅 A 股相关"}
    import akshare as ak
    try:
        df = ak.stock_hsgt_hist_em(symbol="北向资金")
        if not df.empty:
            recent = df.tail(10)
            return {"recent_10d": recent.to_dict("records")}
        return {"note": "北向资金数据为空"}
    except Exception as e:
        return {"error": f"akshare 北向资金失败：{repr(e)[:150]}"}


def _h_sector(symbol: str, market: str) -> dict[str, Any]:
    """板块归属：A股东财，港美无。"""
    if market != "A":
        return {"note": "板块归属仅 A 股有"}
    import akshare as ak
    bare = symbol.split(".")[0]
    out: dict[str, Any] = {}
    # 先查个股自身的行业归属（此前只返回了"全市场有多少个板块"，对个股研究没用）
    try:
        info = ak.stock_individual_info_em(symbol=bare)
        kv = dict(zip(info.iloc[:, 0], info.iloc[:, 1].astype(str), strict=False))
        industry = kv.get("行业")
        if industry:
            out["industry"] = industry
    except Exception:
        pass
    # 再补该行业当日的板块表现（涨跌幅/成交额），判断是不是板块整体在动
    try:
        boards = ak.stock_board_industry_name_em()
        out["industries_total"] = len(boards)
        if out.get("industry") is not None and "板块名称" in boards.columns:
            row = boards[boards["板块名称"] == out["industry"]]
            if not row.empty:
                out["industry_today"] = row.iloc[0].to_dict()
    except Exception as e:
        if not out:
            return {"error": f"akshare 板块失败：{repr(e)[:150]}"}
    if not out:
        return {"note": "未查到板块归属"}
    return out


_PROFILE_KEYS = [
    ("longName", "名称"), ("sector", "板块"), ("industry", "行业"),
    ("country", "国家"), ("city", "城市"), ("fullTimeEmployees", "员工数"),
    ("website", "官网"), ("currency", "计价币种"),
]


def _h_profile(symbol: str, market: str) -> dict[str, Any]:
    """公司画像。A 股走 akshare 个股信息，港美走 yfinance ``info``（v0.34 落地）。"""
    if market == "A":
        import akshare as ak
        try:
            df = ak.stock_individual_info_em(symbol=symbol.split(".")[0])
            kv = dict(zip(df.iloc[:, 0], df.iloc[:, 1].astype(str), strict=False))
            return {"profile": {k: v for k, v in kv.items() if v}}
        except Exception as e:
            return {"error": f"akshare 个股信息失败：{repr(e)[:150]}"}
    try:
        info = _yf_ticker(symbol, market).info or {}
    except Exception as e:
        return {"note": f"yfinance 公司信息拉取失败：{repr(e)[:120]}"}
    prof = {label: info[key] for key, label in _PROFILE_KEYS if info.get(key) is not None}
    summary = info.get("longBusinessSummary")
    if summary:
        prof["主营"] = summary[:400] + ("…" if len(summary) > 400 else "")
    return {"profile": prof} if prof else {"note": "yfinance 未返回公司信息"}


def _h_holders(symbol: str, market: str) -> dict[str, Any]:
    """机构/大股东持股（港美，v0.34 新增）。A 股用 shareholders 板块看股东户数。"""
    if market == "A":
        return {"note": "A 股请看 shareholders（股东户数）板块"}
    t = _yf_ticker(symbol, market)
    out: dict[str, Any] = {}
    try:
        mh = t.major_holders
        if mh is not None and not mh.empty:
            # yfinance 返回单列 DataFrame（索引=指标名），转成 {指标: 值}
            col = mh.columns[0]
            out["major"] = {str(i): mh.loc[i, col] for i in mh.index}
    except Exception:
        pass
    try:
        ih = t.institutional_holders
        if ih is not None and not ih.empty:
            out["institutional"] = ih.head(8).to_dict("records")
    except Exception:
        pass
    return out or {"note": "yfinance 未返回持股数据"}


def _h_sec_filings(symbol: str, market: str) -> dict[str, Any]:
    """SEC 公告：美股，yfinance ``sec_filings``（v0.34 落地）。"""
    if market != "US":
        return {"note": "SEC 公告仅美股有"}
    try:
        fl = _yf_ticker(symbol, market).sec_filings
    except Exception as e:
        return {"note": f"SEC 公告拉取失败：{repr(e)[:120]}"}
    if fl is None or len(fl) == 0:
        return {"note": "yfinance 未返回 SEC 公告"}
    rows = fl.to_dict("records") if hasattr(fl, "to_dict") else list(fl)
    out = []
    for f in rows[:10]:
        if not isinstance(f, dict):
            continue
        out.append({
            "date": str(f.get("date") or f.get("Date") or "")[:10],
            "type": f.get("type") or f.get("Type") or "",
            "title": f.get("title") or f.get("Title") or "",
        })
    return {"filings": out} if out else {"note": "SEC 公告格式无法解析"}


def _h_options(symbol: str, market: str) -> dict[str, Any]:
    """期权链摘要：美股，取最近到期日的看跌/看涨对比（v0.34 落地）。

    只给摘要不给全链——全链几百行，研究报告里翻不动；
    真正有信息量的是 **put/call 未平仓比** 和 **平值隐含波动率**。
    """
    if market != "US":
        return {"note": "期权链仅美股有"}
    t = _yf_ticker(symbol, market)
    try:
        expiries = list(t.options or [])
    except Exception as e:
        return {"note": f"期权到期日拉取失败：{repr(e)[:120]}"}
    if not expiries:
        return {"note": "该标的无期权"}
    exp = expiries[0]
    try:
        chain = t.option_chain(exp)
        calls, puts = chain.calls, chain.puts
    except Exception as e:
        return {"note": f"期权链拉取失败：{repr(e)[:120]}"}
    out: dict[str, Any] = {"expiry": exp, "expiries_total": len(expiries)}
    try:
        c_oi = float(calls["openInterest"].fillna(0).sum())
        p_oi = float(puts["openInterest"].fillna(0).sum())
        out["call_oi"], out["put_oi"] = c_oi, p_oi
        out["put_call_oi"] = round(p_oi / c_oi, 3) if c_oi else None
    except Exception:
        pass
    try:  # 平值 IV：用现价找最接近的行权价
        spot = float(get_recent_bars(symbol, days=5)["close"].iloc[-1])
        atm = calls.iloc[(calls["strike"] - spot).abs().argsort().iloc[0]]
        out["spot"] = round(spot, 2)
        out["atm_strike"] = float(atm["strike"])
        out["atm_call_iv"] = round(float(atm["impliedVolatility"]), 4)
    except Exception:
        pass
    return out


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
    "holders": _h_holders,
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
        elif isinstance(data, dict) and "note" in data:
            # v0.34：原来这里是「建议用外部工具补全」，现在只在数据源真的
            # 没有这项数据时出现，说明为什么没有而不是让用户去找别的工具
            lines.append(f"  ⓘ {data['note']}")
        elif isinstance(data, dict) and "profile" in data:
            for k, v in list(data["profile"].items())[:12]:
                lines.append(f"  {k}: {v}")
        elif isinstance(data, dict) and ("metrics" in data or "income_stmt" in data):
            for k, v in (data.get("metrics") or {}).items():
                lines.append(f"  {k}: {v}")
            for period, rows in (data.get("income_stmt") or {}).items():
                vals = "  ".join(f"{k} {v / 1e8:.2f}亿" for k, v in rows.items()
                                 if isinstance(v, (int, float)))
                lines.append(f"  {period[:10]}  {vals}")
        elif isinstance(data, dict) and ("targets" in data or "ratings" in data):
            t = data.get("targets") or {}
            if t:
                lines.append("  目标价：" + "  ".join(f"{k} {v}" for k, v in t.items()))
            for r in (data.get("ratings") or [])[:3]:
                if isinstance(r, dict):
                    lines.append("  评级：" + "  ".join(f"{k}={v}" for k, v in r.items()))
        elif isinstance(data, dict) and "filings" in data:
            for f in data["filings"][:8]:
                lines.append(f"  {f.get('date', '')}  {f.get('type', '')}  {f.get('title', '')[:60]}")
        elif isinstance(data, dict) and ("major" in data or "institutional" in data):
            for k, v in (data.get("major") or {}).items():
                lines.append(f"  {k}: {v}")
            for h in (data.get("institutional") or [])[:5]:
                if isinstance(h, dict):
                    lines.append(f"  • {h.get('Holder', '')}  {h.get('Shares', '')}"
                                 f"  ({h.get('pctHeld', '')})")
        elif isinstance(data, dict) and "expiry" in data:
            lines.append(f"  最近到期 {data['expiry']}（共 {data.get('expiries_total', '?')} 个到期日）")
            if data.get("put_call_oi") is not None:
                lines.append(f"  未平仓 put/call = {data['put_call_oi']}"
                             f"（put {data.get('put_oi', 0):.0f} / call {data.get('call_oi', 0):.0f}）")
            if data.get("atm_call_iv") is not None:
                lines.append(f"  现价 {data.get('spot')}  平值行权价 {data['atm_strike']}"
                             f"  平值 call 隐波 {data['atm_call_iv']:.1%}")
        elif isinstance(data, dict) and "snapshot" in data:
            # snapshot 板块特殊
            snap = data["snapshot"]
            # 涨跌幅限两位小数：不加精度会打出 -2.3809523809523734%
            lines.append(f"  最新价 {snap.get('close', '?')}  涨跌幅 {snap.get('change_pct', 0):+.2f}%")
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
    "research": "研报/分析师",
    "block_trades": "大宗交易",
    "margin": "融资融券",
    "shareholders": "股东户数",
    "lockup": "解禁",
    "northbound": "北向资金",
    "sector": "板块归属",
    "profile": "公司画像",
    "holders": "机构持股",
    "sec_filings": "SEC 公告",
    "options": "期权链",
}
