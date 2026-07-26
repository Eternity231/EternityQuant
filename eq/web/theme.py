"""仪表盘主题定制（v0.33 新增）—— 看板娘 + 自动配色。

**设计目标**

用户想要「加二次元图」这类个性化，但硬编码某张图的路径进仓库是错的。
所以做成配置驱动：

- 图片路径放 ``.eternityquant/.env`` 的 ``EQ_DASH_IMAGE``（或 ``eq dash --image``）
- **主题色自动从图片提取**（PIL 量化取主色调）——换一张图，
  整个仪表盘的配色跟着换，不用手调任何颜色
- 页面背景 = 图片 + 半透明遮罩（透明度可调，保证表格可读）
- 侧边栏顶部 = 看板娘卡片
- 指标卡片 = 毛玻璃效果，浮在背景上

**性能**：原图可能好几 MB，直接 base64 嵌进页面会让每次刷新都很重。
所以先缩放 + 重压缩（背景 ≤1280px、侧边栏 ≤420px），并按
``(路径, mtime)`` 缓存——图片文件变了缓存自动失效。

环境变量（都可选）：

=====================  ==========================================
``EQ_DASH_IMAGE``       图片路径。不设 = 默认素色主题
``EQ_DASH_OPACITY``     背景遮罩不透明度 0~1，默认 0.88（越大背景越淡）
``EQ_DASH_MASCOT``      侧边栏看板娘开关，默认 on
``EQ_DASH_PRIMARY``     手动指定主色（#rrggbb），不设则从图片提取
=====================  ==========================================
"""

from __future__ import annotations

import base64
import io
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_OPACITY = 0.88
_BG_MAX_W = 1280       # 背景图最长边
_MASCOT_MAX_W = 420    # 侧边栏看板娘宽度
_JPEG_QUALITY = 78


# ======================================================================
# 配置
# ======================================================================

@dataclass
class UIConfig:
    image: Path | None = None
    opacity: float = DEFAULT_OPACITY
    mascot: bool = True
    primary: str | None = None      # 手动主色；None = 从图提取

    @property
    def enabled(self) -> bool:
        return self.image is not None and self.image.exists()


def load_ui_config() -> UIConfig:
    """从环境变量（含 ``.eternityquant/.env``）读 UI 配置。"""
    from eq.core.python_dotenv_loader import load_dotenv_if_present

    load_dotenv_if_present()
    raw = (os.getenv("EQ_DASH_IMAGE") or "").strip().strip('"').strip("'")
    image = Path(raw) if raw else None
    if image is not None and not image.exists():
        logger.warning("EQ_DASH_IMAGE 指向的文件不存在：%s", image)
        image = None
    try:
        opacity = float(os.getenv("EQ_DASH_OPACITY", DEFAULT_OPACITY))
    except ValueError:
        opacity = DEFAULT_OPACITY
    opacity = min(max(opacity, 0.0), 1.0)
    mascot = (os.getenv("EQ_DASH_MASCOT", "on").strip().lower()
              not in ("off", "0", "false", "no"))
    primary = (os.getenv("EQ_DASH_PRIMARY") or "").strip() or None
    if primary and not _valid_hex(primary):
        logger.warning("EQ_DASH_PRIMARY 不是合法颜色：%s（应为 #rrggbb）", primary)
        primary = None
    return UIConfig(image=image, opacity=opacity, mascot=mascot, primary=primary)


def _valid_hex(c: str) -> bool:
    c = c.strip()
    if not (c.startswith("#") and len(c) == 7):
        return False
    try:
        int(c[1:], 16)
        return True
    except ValueError:
        return False


def save_image_to_env(image_path: str | Path) -> Path:
    """把图片路径写进 ``.eternityquant/.env``（已有则替换该行）。"""
    from eq.db import DEFAULT_HOME

    p = Path(str(image_path).strip().strip('"').strip("'"))
    if not p.exists():
        raise FileNotFoundError(f"图片不存在：{p}")
    env_file = DEFAULT_HOME / ".env"
    env_file.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if env_file.exists():
        lines = [ln for ln in env_file.read_text(encoding="utf-8").splitlines()
                 if not ln.strip().startswith("EQ_DASH_IMAGE=")]
    lines.append(f"EQ_DASH_IMAGE={p}")
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env_file


# ======================================================================
# 图片处理（缩放 + base64，按 mtime 缓存）
# ======================================================================

@lru_cache(maxsize=8)
def _encoded(path_str: str, mtime: float, max_w: int) -> str | None:
    """缩放 + JPEG 重压缩 + base64。mtime 进 key：文件变了缓存自动失效。"""
    try:
        from PIL import Image

        im = Image.open(path_str).convert("RGB")
        if max(im.size) > max_w:
            im.thumbnail((max_w, max_w), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        logger.warning("图片处理失败 %s：%s", path_str, e)
        return None


def image_data_uri(path: Path, max_w: int = _BG_MAX_W) -> str | None:
    """图片 → ``data:image/jpeg;base64,...``。失败返回 None（主题静默降级）。"""
    if not path.exists():
        return None
    b64 = _encoded(str(path), path.stat().st_mtime, max_w)
    return f"data:image/jpeg;base64,{b64}" if b64 else None


# ======================================================================
# 自动配色：从图片提取主色调
# ======================================================================

@lru_cache(maxsize=8)
def _palette_cached(path_str: str, mtime: float) -> tuple[tuple[int, int, int], ...]:
    from PIL import Image

    im = Image.open(path_str).convert("RGB")
    im.thumbnail((160, 160))
    q = im.quantize(colors=8, method=Image.Quantize.MEDIANCUT).convert("RGB")
    counts = q.getcolors(160 * 160) or []
    counts.sort(reverse=True)
    return tuple(rgb for _, rgb in counts)


def _luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = (c / 255 for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _saturation(rgb: tuple[int, int, int]) -> float:
    mx, mn = max(rgb) / 255, min(rgb) / 255
    return (mx - mn) / mx if mx else 0.0


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _mix(rgb: tuple[int, int, int], other: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(a + (b - a) * t) for a, b in zip(rgb, other, strict=True))  # type: ignore[return-value]


def extract_palette(path: Path) -> dict[str, Any]:
    """从图片提取一套可用的主题色。

    - ``dominant``：出现最多的颜色 → 背景遮罩的底色（和图片天然协调）
    - ``accent``：主色调里**饱和度最高且明度适中**的 → 强调色（标题/按钮）。
      饱和度过滤很重要：出现最多的往往是灰蒙蒙的背景色，直接用会很脏
    - ``is_light``：图片整体偏亮还是偏暗 → 决定文字用深色还是浅色
    """
    try:
        colors = _palette_cached(str(path), path.stat().st_mtime)
    except Exception as e:
        logger.warning("取色失败 %s：%s", path, e)
        return {"dominant": "#f5f0ea", "accent": "#b08968", "is_light": True,
                "overlay": "#faf7f2"}
    if not colors:
        return {"dominant": "#f5f0ea", "accent": "#b08968", "is_light": True,
                "overlay": "#faf7f2"}

    dominant = colors[0]
    # accent：饱和度优先，且明度别太极端（太黑太白都当不了强调色）
    candidates = [c for c in colors if 0.18 < _luminance(c) < 0.85]
    accent = max(candidates or colors, key=_saturation)
    if _saturation(accent) < 0.08:
        # 整张图接近灰阶：把 dominant 往暖处偏一点当强调色，避免全灰
        accent = _mix(dominant, (176, 137, 104), 0.6)

    is_light = _luminance(dominant) > 0.5
    # 遮罩底色：dominant 向白（亮图）或向黑（暗图）大幅混合，保证正文可读
    overlay = _mix(dominant, (255, 255, 255) if is_light else (17, 17, 17), 0.82)
    # 强调色用于文字时加深/提亮到可读区间
    accent_text = _mix(accent, (0, 0, 0), 0.35) if is_light else _mix(accent, (255, 255, 255), 0.35)
    return {
        "dominant": _hex(dominant),
        "accent": _hex(accent),
        "accent_text": _hex(accent_text),
        "overlay": _hex(overlay),
        "is_light": is_light,
    }


# ======================================================================
# CSS 生成与注入
# ======================================================================

def streamlit_theme_flags(cfg: UIConfig | None = None) -> list[str]:
    """把提取出的配色转成 ``streamlit run`` 的 ``--theme.*`` 命令行参数。

    **为什么不用 CSS 硬覆盖**：Streamlit 的指标数值、下拉框、
    ``st.dataframe``（canvas 渲染）等控件的文字颜色是它自己管的，
    只改容器背景会出现「深色底 + 深色字」看不见的情况（暗色图上实测过）。
    把配色交给 Streamlit 原生主题，所有控件才会一致。

    CSS 那层只负责它做不到的事：背景图、看板娘、毛玻璃卡片。
    """
    cfg = cfg or load_ui_config()
    if not cfg.enabled:
        return []
    try:
        pal = extract_palette(cfg.image)
    except Exception:
        return []
    accent = cfg.primary or pal["accent"]
    is_light = pal["is_light"]
    # 正文背景用遮罩色；控件背景（卡片/输入框/表格）统一往白色挪一档，
    # 这样它们能从背景里"浮"出来——亮色下是更白的卡片，暗色下是更浅的灰块。
    bg = pal["overlay"]
    rgb = tuple(int(bg[i:i + 2], 16) for i in (1, 3, 5))
    second = _hex(_mix(rgb, (255, 255, 255), 0.14 if is_light else 0.10))
    text = "#26221e" if is_light else "#ece8e3"
    return [
        "--theme.base", "light" if is_light else "dark",
        "--theme.primaryColor", accent,
        "--theme.backgroundColor", bg,
        "--theme.secondaryBackgroundColor", second,
        "--theme.textColor", text,
    ]


def build_css(cfg: UIConfig) -> str:
    """按配置生成整套 CSS。没配图片时返回空串（保持默认外观）。"""
    if not cfg.enabled:
        return ""
    bg = image_data_uri(cfg.image, _BG_MAX_W)
    if not bg:
        return ""
    pal = extract_palette(cfg.image)
    accent = cfg.primary or pal["accent"]
    accent_text = cfg.primary or pal.get("accent_text", accent)
    overlay = pal["overlay"]
    op = cfg.opacity
    text = "#2b2620" if pal["is_light"] else "#f0ece6"
    card_bg = "rgba(255,255,255,0.62)" if pal["is_light"] else "rgba(20,20,24,0.62)"

    def rgba(hex_color: str, a: float) -> str:
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        return f"rgba({r},{g},{b},{a})"

    return f"""
<style>
/* ===== EternityQuant 主题（由 {cfg.image.name} 自动配色）===== */
[data-testid="stAppViewContainer"] {{
    background:
        linear-gradient({rgba(overlay, op)}, {rgba(overlay, op)}),
        url("{bg}") center / cover fixed no-repeat;
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] > div:first-child {{
    background: {rgba(overlay, min(op + 0.06, 1.0))};
    border-right: 1px solid {rgba(accent, 0.25)};
}}
h1, h2, h3 {{ color: {accent_text} !important; }}
[data-testid="stMetric"] {{
    background: {card_bg};
    border: 1px solid {rgba(accent, 0.22)};
    border-radius: 14px;
    padding: 12px 16px;
    backdrop-filter: blur(6px);
    box-shadow: 0 2px 10px {rgba(accent, 0.10)};
}}
[data-testid="stMetricLabel"] {{ color: {text}; opacity: 0.75; }}
[data-testid="stDataFrame"], [data-testid="stTable"] {{
    background: {card_bg};
    border-radius: 12px;
    backdrop-filter: blur(6px);
}}
[data-testid="stExpander"] {{
    background: {card_bg};
    border-radius: 12px;
}}
.stButton > button {{
    border: 1px solid {rgba(accent, 0.55)};
    border-radius: 10px;
}}
.stButton > button:hover {{
    border-color: {accent};
    color: {accent_text};
}}
.eq-mascot img {{
    border-radius: 14px;
    box-shadow: 0 4px 14px {rgba(accent, 0.30)};
}}
</style>
"""


def apply(st) -> dict[str, Any]:
    """在 Streamlit 页面里应用主题。返回配置与配色（侧边栏看板娘用）。

    任何一步失败都静默降级为默认外观——主题坏了不能拖垮仪表盘。
    """
    info: dict[str, Any] = {"enabled": False}
    try:
        cfg = load_ui_config()
        if not cfg.enabled:
            return info
        css = build_css(cfg)
        if css:
            st.markdown(css, unsafe_allow_html=True)
        info = {"enabled": True, "cfg": cfg, "palette": extract_palette(cfg.image)}
        if cfg.mascot:
            uri = image_data_uri(cfg.image, _MASCOT_MAX_W)
            if uri:
                with st.sidebar:
                    st.markdown(
                        f'<div class="eq-mascot"><img src="{uri}" '
                        f'style="width:100%"/></div>',
                        unsafe_allow_html=True,
                    )
    except Exception as e:
        logger.warning("主题应用失败，回退默认外观：%s", e)
    return info
