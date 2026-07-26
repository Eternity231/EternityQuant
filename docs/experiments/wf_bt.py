"""滚动重训的预测能不能变成钱。

wf.py 只回答了「IC 有没有提升」（h=5：+0.0135 → +0.0222）。
IC 是抽象数字，换手和成本会不会把这点改善吃光，得跑组合回测才知道。
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("wfbt")
log.setLevel(logging.INFO)

from eq.backtest.portfolio import PortfolioConfig  # noqa: E402
from eq.cli import _resolve_symbols  # noqa: E402
from eq.strategy.factors import local_train as lt  # noqa: E402

OUT = Path(__file__).with_name("wf_bt_results.json")
UNIVERSE = "file:.eternityquant/ml_universe.txt"
HORIZONS = [5, 10, 20]
REBALANCES = ["monthly", "weekly"]
POSITIONS = [10, 20]

out: dict = {"runs": [], "errors": []}


def save():
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")


def main():
    symbols, label = _resolve_symbols(UNIVERSE)
    log.info("股票池 %s → %d 只", label, len(symbols))
    bars = lt.load_bars(symbols, days=1200)
    log.info("行情 %d 只", len(bars))

    for h in HORIZONS:
        try:
            t0 = time.perf_counter()
            wf = lt.walk_forward_local(symbols, algo="lightgbm", horizon=h,
                                       n_folds=6, test_days=40, n_seeds=1)
            o = wf["oos"]
            log.info("[h=%d] 滚动 OOS IC %+.4f  t_nw %+.2f  (%ds)", h,
                     o["ic_mean"], o["t_stat_nw"], round(time.perf_counter() - t0))
        except Exception as e:
            out["errors"].append({"stage": "wf", "h": h, "err": repr(e)[:300],
                                  "tb": traceback.format_exc()[-1200:]})
            log.error("[h=%d] 滚动重训失败：%s", h, repr(e)[:200])
            save()
            continue

        for reb in REBALANCES:
            for pos in POSITIONS:
                rec = {"horizon": h, "rebalance": reb, "positions": pos,
                       "oos_ic": o["ic_mean"], "oos_t_nw": o["t_stat_nw"],
                       "oos_days": o["n_days"]}
                try:
                    cfg = PortfolioConfig(max_positions=pos, rebalance=reb,
                                          allocation="score", cost_model="a_share")
                    bt = lt.backtest_predictions(wf["predictions"], bars,
                                                 top_n=pos, cfg=cfg)
                    m = bt["result"].metrics
                    rec.update({
                        "net": m.get("total_return"),
                        "gross": bt["gross"].metrics.get("total_return"),
                        "benchmark": bt["benchmark_return"],
                        "excess": bt["excess_return"],
                        "cost_drag": bt["cost_drag"],
                        "turnover": m.get("annual_turnover"),
                        "max_dd": m.get("max_drawdown"),
                        "start": bt["start"], "bt_days": bt["n_days"],
                        "ok": True,
                    })
                    log.info("  [h=%d %s pos=%d] 净 %+.2f%% 基准 %+.2f%% "
                             "**超额 %+.2f%%** 换手 %.0fx", h, reb, pos,
                             100 * rec["net"], 100 * rec["benchmark"],
                             100 * rec["excess"], rec["turnover"] or 0)
                except Exception as e:
                    rec.update({"ok": False, "err": repr(e)[:300]})
                    out["errors"].append({"stage": "bt", "h": h, "reb": reb,
                                          "pos": pos, "err": repr(e)[:300],
                                          "tb": traceback.format_exc()[-1200:]})
                    log.error("  [h=%d %s pos=%d] 回测失败：%s", h, reb, pos,
                              repr(e)[:200])
                out["runs"].append(rec)
                save()

    save()
    ok = [r for r in out["runs"] if r.get("ok")]
    if ok:
        df = pd.DataFrame(ok)[["horizon", "rebalance", "positions", "oos_ic",
                               "oos_t_nw", "net", "benchmark", "excess",
                               "cost_drag", "turnover"]]
        print("\n" + df.to_string(index=False))
        print(f"\n超额为正：{sum(1 for r in ok if r['excess'] > 0)}/{len(ok)}")
    log.info("滚动重训回测完成：%d 格，%d 个错误", len(out["runs"]), len(out["errors"]))


if __name__ == "__main__":
    main()
