""".env 加载助手（不引入 python-dotenv 重依赖就解决 key=value 解析）。

PROJECT_ROOT 基准目录，供 db.py/DEFAULT_HOME 等共用。"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ENV = PROJECT_ROOT / ".eternityquant" / ".env"

# 统一 home 根目录：始终用项目内 .eternityquant/，避免数据散落到 ~/.eternityquant/
DEFAULT_HOME = PROJECT_ROOT / ".eternityquant"


def load_dotenv_if_present(path: Path | None = None) -> None:
    """加载 .env 文件到 os.environ，已存在的 key 不覆盖。文件不存在静默返回。"""
    path = path or DEFAULT_ENV
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)
