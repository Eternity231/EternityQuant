""".env 加载助手（不引入 python-dotenv 重依赖就解决 key=value 解析）。"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ENV = Path.home() / ".eternityquant" / ".env"


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
