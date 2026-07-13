"""Streamlit 仪表盘子包。

- dashboard.py：主入口，`eq dash` 启动
- sections/：各页块（持仓 / 自选 / 信号 / 回测）
"""

from eq.web.dashboard import run_dashboard

__all__ = ["run_dashboard"]
