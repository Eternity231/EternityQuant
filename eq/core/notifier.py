"""推送通道抽象 + 实现（problem 12 冶议）。

第一版两个实现：
- WeChatWorkBot：企业微信群机器人 webhook（markdown 富文本）
- DesktopNotifier：桌面 toast + 终端 BEL（人在电脑前的零打扰兜底）

通道由规则的 channels JSON 列表决定，多通道并发发送。
配置走 ~/.eternityquant/.env：
- WECHAT_WORK_WEBHOOK：企业微信群机器人完整 webhook URL
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from typing import Any

from eq.core.python_dotenv_loader import load_dotenv_if_present


class Notifier(ABC):
    """推送通道抽象。"""

    name: str

    @abstractmethod
    def send(self, title: str, body: str, **kwargs: Any) -> bool:
        """发送一条通知。返回是否成功。title 用于主题行/标题栏，body 用于正文（markdown）。"""
        ...


class WeChatWorkBot(Notifier):
    """企业微信群机器人 webhook。markdown 正文。"""

    name = "wechat_work"

    def __init__(self, webhook: str | None = None) -> None:
        self.webhook = webhook or os.getenv("WECHAT_WORK_WEBHOOK")
        if not self.webhook:
            raise RuntimeError(
                "WECHAT_WORK_WEBHOOK 未配置：在 .eternityquant/.env 写 WECHAT_WORK_WEBHOOK=https://..."
            )

    def send(self, title: str, body: str, **kwargs: Any) -> bool:
        # 企业微信 markdown 格式：用 <font color="info|comment|warning">...</font> 着色
        content = f"### {title}\n\n{body}"
        payload = {"msgtype": "markdown", "markdown": {"content": content}}
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.webhook, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
            return resp_data.get("errcode") == 0
        except (urllib.error.URLError, json.JSONDecodeError, KeyError) as e:
            print(f"[WeChatWorkBot] 推送失败：{e}", file=sys.stderr)
            return False


class DesktopNotifier(Notifier):
    """桌面通知。Windows toast（PowerShell）+ 终端 BEL 兜底。"""

    name = "desktop"

    def send(self, title: str, body: str, **kwargs: Any) -> bool:
        # 终端 BEL（tty 上闪/响）
        try:
            sys.stdout.write("\a")
            sys.stdout.flush()
        except Exception:
            pass
        # Windows toast via PowerShell BurntToast 模块（若装了）
        if sys.platform == "win32":
            try:
                ps_cmd = (
                    "New-BurntToastNotification -Text "
                    f"'{title.replace(chr(39), chr(96))}', '{body.replace(chr(39), chr(96))[:200]}'"
                )
                subprocess.run(
                    ["powershell", "-NoProfile", "-Command", ps_cmd],
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
        # stdout 兜底（始终执行，让用户在终端里至少看到内容）
        print(f"\n[桌面通知] {title}\n{body}\n")
        return True


_REGISTRY: dict[str, Notifier] | None = None


def _registry() -> dict[str, Notifier]:
    global _REGISTRY
    if _REGISTRY is None:
        load_dotenv_if_present()
        _REGISTRY = {}
        # 桌面通知始终注册（无配置门槛）
        _REGISTRY["desktop"] = DesktopNotifier()
        # 企业微信按需注册（没 webhook 不入注册表，规则里要用会拊错）
        if os.getenv("WECHAT_WORK_WEBHOOK"):
            try:
                _REGISTRY["wechat_work"] = WeChatWorkBot()
            except RuntimeError:
                pass
    return _REGISTRY


def get_channel(name: str) -> Notifier:
    """按名取通道。未注册则拊错。"""
    reg = _registry()
    if name not in reg:
        raise KeyError(f"通道 {name} 未注册（可能未配置环境变量）")
    return reg[name]


def available_channels() -> list[str]:
    """列出当前已注册的通道名。"""
    return list(_registry().keys())


def dispatch(channels: list[str], title: str, body: str, **kwargs: Any) -> dict[str, bool]:
    """并发推送到多个通道。返回 {channel_name: success}。"""
    results: dict[str, bool] = {}
    for ch in channels:
        try:
            notifier = get_channel(ch)
            results[ch] = notifier.send(title, body, **kwargs)
        except (KeyError, RuntimeError) as e:
            results[ch] = False
            print(f"[notifier] 通道 {ch} 不可用：{e}", file=sys.stderr)
    return results
