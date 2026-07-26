"""Streamlit 仪表盘主入口。

`eq dash` 命令通过 subprocess 启动 `streamlit run dashboard.py`，
本模块提供 run_dashboard() 启动器，避免 streamlit 进程内嵌跑出怪问题。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).parent
_ENTRY = _HERE / "dashboard.py"


def run_dashboard(port: int = 8501) -> int:
    """启动 streamlit 网页仪表盘。返回子进程退出码。"""
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(_ENTRY),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    # 主题配色交给 Streamlit 原生主题系统（--theme.*），页面里的 CSS
    # 只负责背景图/看板娘/毛玻璃。只用 CSS 的话，指标数值、下拉框、
    # st.dataframe 这些控件的文字颜色不受控，暗色图上会出现深底深字看不见。
    try:
        from eq.web.theme import streamlit_theme_flags

        cmd += streamlit_theme_flags()
    except Exception:
        pass  # 主题失败不该阻止仪表盘启动
    print(f"启动 Streamlit 仪表盘：http://localhost:{port}")
    try:
        return subprocess.run(cmd).returncode
    except KeyboardInterrupt:
        return 0
