"""把 grid_results.json 变成一张对照表 + 一个明确结论。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SRC = Path(__file__).with_name("grid_results.json")


def main():
    data = json.loads(SRC.read_text(encoding="utf-8"))
    rows = [r for r in data["results"] if r.get("ok")]
    errs = data.get("errors", [])
    if not rows:
        print("没有成功的格子。错误：")
        for e in errs:
            print(" ", e.get("stage"), e.get("h"), e.get("err"))
        return

    df = pd.DataFrame(rows)

    # ---------- 0. 补算等权买入持有基准 ----------
    # 网格跑起来时代码里还没有基准；基准只取决于「哪批票 + 哪段时间」，
    # 事后按每个 start 算一次贴回去即可，不用重跑网格。
    #
    # 这一步是**结论成立的前提**：纯多头组合的收益里天然混着大盘 beta，
    # 大盘涨的时候随便选十只都赚钱，那不是模型的功劳。
    from eq.cli import _resolve_symbols
    from eq.strategy.factors.local_train import load_bars

    syms, _ = _resolve_symbols("file:.eternityquant/ml_universe.txt")
    bars = load_bars(syms, days=1200)
    bench_cache: dict[str, float] = {}
    for st in df["start"].dropna().unique():
        lo = pd.Timestamp(st)
        rs = []
        for d in bars.values():
            c = d["close"][d.index >= lo].dropna()
            if len(c) >= 2 and float(c.iloc[0]) > 0:
                rs.append(float(c.iloc[-1]) / float(c.iloc[0]) - 1.0)
        bench_cache[st] = float(pd.Series(rs).mean()) if rs else 0.0
        print(f"[基准] {st} 起等权买入持有 {len(rs)} 只 → {bench_cache[st]:+.2%}")
    df["benchmark"] = df["start"].map(bench_cache)
    df["excess"] = df["net_return"] - df["benchmark"]

    # ---------- 1. 模型质量（每个 horizon 一个模型，去重）----------
    m = (df.drop_duplicates("horizon")
           .set_index("horizon")[["test_ic", "test_icir", "test_t", "test_t_nw",
                                  "test_win", "n_samples"]])
    m["baseline_ic"] = [df[df.horizon == h]["baseline"].iloc[0].get("ic")
                        if df[df.horizon == h]["baseline"].iloc[0] else None
                        for h in m.index]
    m["baseline_t_nw"] = [df[df.horizon == h]["baseline"].iloc[0].get("t_nw")
                          if df[df.horizon == h]["baseline"].iloc[0] else None
                          for h in m.index]
    print("=" * 78)
    print("一、模型质量（测试段，t_nw 为重叠标签修正后的 t 值）")
    print("=" * 78)
    show = m.copy()
    for c in ("test_ic", "baseline_ic"):
        show[c] = show[c].map(lambda v: f"{v:+.4f}" if pd.notna(v) else "-")
    for c in ("test_icir", "test_t", "test_t_nw", "baseline_t_nw"):
        show[c] = show[c].map(lambda v: f"{v:+.3f}" if pd.notna(v) else "-")
    show["test_win"] = show["test_win"].map(
        lambda v: f"{v:.0%}" if pd.notna(v) else "-")
    show["n_samples"] = show["n_samples"].map(lambda v: f"{v:,}")
    print(show.to_string())

    # ---------- 2. 组合回测 ----------
    print()
    print("=" * 78)
    print("二、组合回测（只在测试段）")
    print("   毛=零成本对照  净=含A股真实成本  基准=同批票等权买入持有  超额=净-基准")
    print("   **只有「超额」为正才说明选股创造了价值**——纯多头的收益里混着大盘 beta")
    print("=" * 78)
    bt = df[["horizon", "rebalance", "positions", "gross_return", "net_return",
             "benchmark", "excess", "cost_drag", "max_dd", "turnover",
             "bt_days"]].copy()
    bt = bt.sort_values(["horizon", "rebalance", "positions"])
    for c in ("gross_return", "net_return", "benchmark", "excess",
              "cost_drag", "max_dd"):
        bt[c] = bt[c].map(lambda v: f"{v:+.2%}" if pd.notna(v) else "-")
    bt["turnover"] = bt["turnover"].map(lambda v: f"{v:.0f}x" if pd.notna(v) else "-")
    print(bt.to_string(index=False))

    # ---------- 3. 结论 ----------
    print()
    print("=" * 78)
    print("三、结论")
    print("=" * 78)
    d = df
    best = d.loc[d["excess"].idxmax()]
    print(f"超额最高的一格：horizon={best['horizon']} {best['rebalance']} "
          f"持仓{best['positions']}只")
    print(f"  净 {best['net_return']:+.2%}  基准 {best['benchmark']:+.2%}  "
          f"→ **超额 {best['excess']:+.2%}**（成本吃 {best['cost_drag']:.2%}）")

    print(f"净收益为正的格子：{int((d['net_return'] > 0).sum())}/{len(d)}")
    print(f"**超额为正**的格子：{int((d['excess'] > 0).sum())}/{len(d)}  ← 这个才算数")
    print(f"净收益跨度：{d['net_return'].min():+.2%} ~ {d['net_return'].max():+.2%}"
          f"（同一批模型换配置的差异；跨度大 = 结果由少数运气票主导，不是能力）")

    sig = m[m["test_t_nw"].abs() >= 2] if "test_t_nw" in m else m.iloc[0:0]
    print("模型 test t_nw 达到 |t|>=2 的 horizon：",
          list(sig.index) if len(sig) else "无")

    # 成本占毛收益的比例
    d = d.assign(cost_share=d["cost_drag"] / d["gross_return"].abs().replace(0, None))
    print(f"成本吃掉毛收益的中位比例：{d['cost_share'].median():.0%}")

    # 换手 vs 成本
    print(f"换手范围 {d['turnover'].min():.0f}x ~ {d['turnover'].max():.0f}x/年")

    if errs:
        print(f"\n中途失败 {len(errs)} 处：")
        for e in errs[:8]:
            print(f"  [{e.get('stage')}] h={e.get('h')} {e.get('err', '')[:120]}")


if __name__ == "__main__":
    main()
