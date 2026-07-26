"""过夜网格实验：horizon × rebalance × positions。

目标是回答一个问题——**日频/周频/月频的 158 因子选股，对散户的成本结构成不成立**。

每格给出：test IC（含重叠修正 t）、无参数基准对照、毛收益、净收益、成本占比。
任何一格失败都只记下来继续，不中断整体。
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("grid")
log.setLevel(logging.INFO)

from eq.backtest.portfolio import PortfolioConfig  # noqa: E402
from eq.cli import _resolve_symbols  # noqa: E402
from eq.strategy.factors import local_train as lt  # noqa: E402

OUT = Path(__file__).with_name("grid_results.json")
UNIVERSE = "file:.eternityquant/ml_universe.txt"
HORIZONS = [5, 10, 20]
REBALANCES = ["weekly", "monthly"]
POSITIONS = [10, 20]
FAST_PARAMS = None          # 用默认（会按样本量自动缩放正则）

results: list[dict] = []
errors: list[dict] = []


def save():
    OUT.write_text(json.dumps({"results": results, "errors": errors},
                              ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")


def main():
    symbols, label = _resolve_symbols(UNIVERSE)
    log.info("股票池 %s → %d 只", label, len(symbols))

    for h in HORIZONS:
        # ---- 1. 无参数基准（每个 horizon 一次）----
        base = None
        try:
            t0 = time.perf_counter()
            b = lt.baseline_composite(symbols, horizon=h, top_k=5)
            c = b["composite"]
            base = {"ic": c["ic_mean"], "icir": c["icir"],
                    "t": c["t_stat"], "t_nw": c["t_stat_nw"],
                    "win": c["ic_win_rate"], "factors": b["selected"],
                    "n_test_days": b["n_test_days"]}
            log.info("[h=%d] 基准 IC %+.4f  t_nw %+.2f  因子 %s  (%.0fs)",
                     h, base["ic"], base["t_nw"], ",".join(b["selected"][:3]),
                     time.perf_counter() - t0)
        except Exception as e:
            errors.append({"stage": "baseline", "h": h, "err": repr(e)[:300],
                           "tb": traceback.format_exc()[-1500:]})
            log.error("[h=%d] 基准失败：%s", h, repr(e)[:200])
        save()

        # ---- 2. 训练（每个 horizon 一个模型，3 种子集成）----
        try:
            t0 = time.perf_counter()
            r = lt.train_local(symbols, algo="lightgbm", horizon=h, n_seeds=3,
                               params=FAST_PARAMS, universe_label="ml520",
                               name=f"grid_h{h}")
            tr = r["metrics"]["test"] or {}
            log.info("[h=%d] 训练完成 %s  test IC %+.4f  t_nw %+.2f  (%.0fs)",
                     h, r["model_id"], r["metrics"]["ic"],
                     tr.get("t_stat_nw", 0), time.perf_counter() - t0)
            if r["diagnosis"]:
                log.warning("[h=%d] 自检告警：%s", h, r["diagnosis"])
        except Exception as e:
            errors.append({"stage": "train", "h": h, "err": repr(e)[:300],
                           "tb": traceback.format_exc()[-1500:]})
            log.error("[h=%d] 训练失败：%s", h, repr(e)[:200])
            save()
            continue

        # ---- 3. 回测（同一模型，不同调仓节奏/持仓数）----
        for reb in REBALANCES:
            for pos in POSITIONS:
                cell = {"horizon": h, "rebalance": reb, "positions": pos,
                        "model_id": r["model_id"],
                        "test_ic": r["metrics"]["ic"],
                        "test_icir": tr.get("icir"),
                        "test_t": tr.get("t_stat"),
                        "test_t_nw": tr.get("t_stat_nw"),
                        "test_win": tr.get("ic_win_rate"),
                        "n_symbols": r["n_symbols"],
                        "n_samples": r["n_samples"],
                        "diagnosis": r["diagnosis"],
                        "baseline": base}
                try:
                    t0 = time.perf_counter()
                    cfg = PortfolioConfig(max_positions=pos, rebalance=reb,
                                          allocation="score", cost_model="a_share")
                    bt = lt.backtest_local(r["model_id"], symbols, top_n=pos, cfg=cfg)
                    m, g = bt["result"].metrics, bt["gross"].metrics
                    cell.update({
                        "start": bt["start"], "bt_days": bt["n_days"],
                        "net_return": m.get("total_return"),
                        "gross_return": g.get("total_return"),
                        "cost_drag": bt["cost_drag"],
                        "annual_return": m.get("annual_return"),
                        "sharpe": m.get("sharpe"),
                        "max_dd": m.get("max_drawdown"),
                        "turnover": m.get("annual_turnover"),
                        "n_trades": m.get("num_trades"),
                        "ok": True,
                    })
                    log.info("  [h=%d %s pos=%d] 毛 %+.2f%% 净 %+.2f%% 成本 %.2f%% "
                             "换手 %.0fx (%.0fs)", h, reb, pos,
                             100 * g.get("total_return", 0), 100 * m.get("total_return", 0),
                             100 * bt["cost_drag"], m.get("annual_turnover", 0),
                             time.perf_counter() - t0)
                except Exception as e:
                    cell.update({"ok": False, "err": repr(e)[:300]})
                    errors.append({"stage": "backtest", "h": h, "reb": reb,
                                   "pos": pos, "err": repr(e)[:300],
                                   "tb": traceback.format_exc()[-1500:]})
                    log.error("  [h=%d %s pos=%d] 回测失败：%s", h, reb, pos,
                              repr(e)[:200])
                results.append(cell)
                save()

    save()
    log.info("网格完成：%d 格，%d 个错误", len(results), len(errors))
    ok = [r for r in results if r.get("ok")]
    if ok:
        df = pd.DataFrame(ok)[["horizon", "rebalance", "positions", "test_ic",
                               "test_t_nw", "gross_return", "net_return",
                               "cost_drag", "turnover", "max_dd"]]
        print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
