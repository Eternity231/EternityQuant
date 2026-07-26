"""滚动重训对照实验：和单次切分比，样本外表现有没有变好。

网格实验回答的是「这条路赚不赚钱」；这个脚本回答「换个更贴合市场的训练方式
能不能救回来」——这是唯一一个真正改算法的实验，其余都是配置调整。
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("wf")
log.setLevel(logging.INFO)

from eq.cli import _resolve_symbols  # noqa: E402
from eq.strategy.factors import local_train as lt  # noqa: E402

OUT = Path(__file__).with_name("wf_results.json")
UNIVERSE = "file:.eternityquant/ml_universe.txt"
HORIZONS = [5, 10, 20]
FOLDS = 6
TEST_DAYS = 40

out: dict = {"runs": [], "errors": []}


def save():
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")


def main():
    symbols, label = _resolve_symbols(UNIVERSE)
    log.info("股票池 %s → %d 只  滚动 %d 折 x %d 天", label, len(symbols),
             FOLDS, TEST_DAYS)

    for h in HORIZONS:
        try:
            t0 = time.perf_counter()
            r = lt.walk_forward_local(symbols, algo="lightgbm", horizon=h,
                                      n_folds=FOLDS, test_days=TEST_DAYS,
                                      n_seeds=1)
            o = r["oos"]
            ics = [f["ic"] for f in r["folds"]]
            rec = {
                "horizon": h,
                "n_folds_ok": r["n_folds_ok"],
                "n_oos_samples": r["n_oos_samples"],
                "oos_ic": o["ic_mean"], "oos_icir": o["icir"],
                "oos_t": o["t_stat"], "oos_t_nw": o["t_stat_nw"],
                "oos_win": o["ic_win_rate"], "oos_days": o["n_days"],
                "fold_ics": ics,
                "folds_positive": sum(1 for v in ics if v > 0),
                "folds": r["folds"],
                "elapsed_s": round(time.perf_counter() - t0),
            }
            out["runs"].append(rec)
            log.info("[h=%d] 滚动样本外 IC %+.4f  t_nw %+.2f  逐折为正 %d/%d  (%ds)",
                     h, o["ic_mean"], o["t_stat_nw"], rec["folds_positive"],
                     len(ics), rec["elapsed_s"])
        except Exception as e:
            out["errors"].append({"h": h, "err": repr(e)[:300],
                                  "tb": traceback.format_exc()[-1500:]})
            log.error("[h=%d] 滚动重训失败：%s", h, repr(e)[:200])
        save()

    save()
    if out["runs"]:
        df = pd.DataFrame(out["runs"])[
            ["horizon", "n_folds_ok", "n_oos_samples", "oos_days",
             "oos_ic", "oos_icir", "oos_t_nw", "oos_win", "folds_positive"]]
        print("\n" + df.to_string(index=False))
    log.info("滚动重训实验完成：%d 组，%d 个错误", len(out["runs"]), len(out["errors"]))


if __name__ == "__main__":
    main()
