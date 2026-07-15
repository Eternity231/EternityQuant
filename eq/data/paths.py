"""统一数据路径管理。

所有市场数据集中在 ``~/.eternityquant/data/`` 下，按市场分子目录：

::

    .eternityquant/
    ├─ data/
    │  ├─ a/                      # A 股（qlib .bin 格式）
    │  │  └─ qlib_cn_data/        # qlib provider_uri
    │  │     ├─ calendars/
    │  │     ├─ features/
    │  │     ├─ instruments/
    │  │     └─ all_codes.txt
    │  ├─ hk/                     # 港股
    │  │  ├─ daily/               # 日线 CSV（akshare Sina 源）
    │  │  ├─ 5m/                  # 5 分钟线 CSV（yfinance）
    │  │  ├─ 1m/                  # 1 分钟线 CSV（yfinance）
    │  │  ├─ features/            # 计算后的特征 CSV（训练用）
    │  │  └─ models/              # 港股模型 pkl
    │  └─ us/                     # 美股
    │     └─ daily/               # 日线 CSV（yfinance）
    ├─ ml_models/                 # 全局 ML 模型 pkl
    └─ ...

历史目录（已被本模块取代）：
- ``.qlib_data/cn_data``  → ``data/a/qlib_cn_data``
- ``.eternityquant/hk_data`` → ``data/hk``
- ``.eternityquant/us_data`` → ``data/us``

调用 :func:`ensure_data_dirs` 会在第一次访问时自动创建全部目录，
:func:`migrate_legacy_data_layout` 会把旧目录的数据迁移到新位置（迁移后旧目录保留，不破坏现有脚本）。
"""

from __future__ import annotations

import shutil
from pathlib import Path

# ~/.eternityquant 根目录（与 eq.db.DEFAULT_HOME 对齐）
_HOME = Path.home() / ".eternityquant"
# 项目内 .eternityquant 兼容旧脚本（collector.py 之前用这个）
_PROJECT_HOME = Path(__file__).resolve().parent.parent.parent / ".eternityquant"

# 优先使用项目内目录（保持与现有 ml_models / eternityquant.db 一致）
DATA_ROOT = _PROJECT_HOME / "data"

# A 股 qlib 数据
A_DIR = DATA_ROOT / "a"
QLIB_CN_DATA_DIR = A_DIR / "qlib_cn_data"

# 港股数据
HK_DIR = DATA_ROOT / "hk"
HK_DAILY_DIR = HK_DIR / "daily"
HK_5M_DIR = HK_DIR / "5m"
HK_1M_DIR = HK_DIR / "1m"
HK_FEAT_DIR = HK_DIR / "features"
HK_MODELS_DIR = HK_DIR / "models"

# 美股数据
US_DIR = DATA_ROOT / "us"
US_DAILY_DIR = US_DIR / "daily"


def ensure_data_dirs() -> None:
    """创建所有市场数据目录（幂等）。"""
    for d in (
        A_DIR, QLIB_CN_DATA_DIR,
        HK_DIR, HK_DAILY_DIR, HK_5M_DIR, HK_1M_DIR, HK_FEAT_DIR, HK_MODELS_DIR,
        US_DIR, US_DAILY_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


_LEGACY_QLIB_DIR = Path(__file__).resolve().parent.parent.parent.parent / ".qlib_data" / "cn_data"
_LEGACY_HK_DIR = _PROJECT_HOME / "hk_data"
_LEGACY_US_DIR = _PROJECT_HOME / "us_data"


def migrate_legacy_data_layout(*, dry_run: bool = False, verbose: bool = True) -> dict[str, list[str]]:
    """把旧散落目录的数据迁移到统一的 ``data/`` 下。

    迁移规则（仅当目标不存在时才复制，绝不覆盖）：
    - ``.qlib_data/cn_data`` → ``data/a/qlib_cn_data``
    - ``.eternityquant/hk_data/{daily,5m,1m,features,models}`` → ``data/hk/...``
    - ``.eternityquant/us_data/features`` → ``data/us/daily`` （旧版 us 数据直接放 features/）

    Args:
        dry_run: 只打印将要做什么，不真正复制
        verbose: 打印每一步

    Returns:
        {"copied": [...], "skipped": [...]}
    """
    copied: list[str] = []
    skipped: list[str] = []

    def _migrate(src: Path, dst: Path) -> None:
        if not src.exists():
            return
        if dst.exists() and any(dst.iterdir()):
            if verbose:
                print(f"  · 跳过 {src} → {dst}（目标已存在）", flush=True)
            skipped.append(f"{src} → {dst}")
            return
        dst.mkdir(parents=True, exist_ok=True)
        if dry_run:
            if verbose:
                print(f"  · [dry-run] 将复制 {src} → {dst}", flush=True)
            return
        # 逐项复制（merging into existing dst）
        for item in src.iterdir():
            target = dst / item.name
            if target.exists():
                continue
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
        if verbose:
            print(f"  ✓ {src} → {dst}", flush=True)
        copied.append(f"{src} → {dst}")

    ensure_data_dirs()
    if verbose:
        print("迁移旧数据目录到统一 data/ 下：", flush=True)

    # A 股 qlib 数据
    _migrate(_LEGACY_QLIB_DIR, QLIB_CN_DATA_DIR)

    # 港股数据（5 个子目录）
    for sub in ("daily", "5m", "1m", "features", "models"):
        _migrate(_LEGACY_HK_DIR / sub, {
            "daily": HK_DAILY_DIR, "5m": HK_5M_DIR, "1m": HK_1M_DIR,
            "features": HK_FEAT_DIR, "models": HK_MODELS_DIR,
        }[sub])

    # 美股数据（旧版 us_data/features/*.csv → data/us/daily/）
    _migrate(_LEGACY_US_DIR / "features", US_DAILY_DIR)

    return {"copied": copied, "skipped": skipped}
