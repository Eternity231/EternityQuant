"""仪表盘主题定制（v0.33）。用 PIL 造小图测，不依赖用户本地图片。"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from eq.web import theme  # noqa: E402


@pytest.fixture
def warm_img(tmp_path) -> Path:
    """一张暖色调（米棕系）的测试图。"""
    im = Image.new("RGB", (320, 200), (196, 168, 142))
    # 加一块高饱和色，让 accent 提取有东西可选
    for x in range(40, 120):
        for y in range(40, 120):
            im.putpixel((x, y), (168, 96, 52))
    p = tmp_path / "warm.jpg"
    im.save(p, quality=90)
    return p


@pytest.fixture
def dark_img(tmp_path) -> Path:
    im = Image.new("RGB", (200, 200), (24, 26, 34))
    p = tmp_path / "dark.jpg"
    im.save(p)
    return p


@pytest.fixture
def _clean_env(monkeypatch, tmp_path):
    """隔离主题配置：既清环境变量，**也断开真实的 .eternityquant/.env**。

    ``load_ui_config()`` 会调 ``load_dotenv_if_present()``，它默认读
    ``.eternityquant/.env``。只 delenv 不够——用户真配了 EQ_DASH_IMAGE 的话，
    dotenv 会把它塞回环境，"未配置"这组用例就永远测不到。
    把 DEFAULT_ENV 指向一个不存在的临时路径才算真隔离。
    """
    from eq.core import python_dotenv_loader as loader

    monkeypatch.setattr(loader, "DEFAULT_ENV", tmp_path / "nonexistent.env")
    for k in ("EQ_DASH_IMAGE", "EQ_DASH_OPACITY", "EQ_DASH_MASCOT", "EQ_DASH_PRIMARY"):
        monkeypatch.delenv(k, raising=False)


# ====================== 配置 ======================

def test_config_defaults(_clean_env):
    cfg = theme.load_ui_config()
    assert cfg.image is None and not cfg.enabled
    assert cfg.opacity == theme.DEFAULT_OPACITY
    assert cfg.mascot is True


def test_config_from_env(_clean_env, monkeypatch, warm_img):
    monkeypatch.setenv("EQ_DASH_IMAGE", f'"{warm_img}"')     # 带引号也要认
    monkeypatch.setenv("EQ_DASH_OPACITY", "0.5")
    monkeypatch.setenv("EQ_DASH_MASCOT", "off")
    cfg = theme.load_ui_config()
    assert cfg.enabled and cfg.image == warm_img
    assert cfg.opacity == 0.5
    assert cfg.mascot is False


def test_config_missing_file_disables(_clean_env, monkeypatch):
    monkeypatch.setenv("EQ_DASH_IMAGE", r"Z:\不存在\x.jpg")
    assert not theme.load_ui_config().enabled


def test_config_opacity_clamped_and_bad_values(_clean_env, monkeypatch, warm_img):
    monkeypatch.setenv("EQ_DASH_IMAGE", str(warm_img))
    monkeypatch.setenv("EQ_DASH_OPACITY", "7")
    assert theme.load_ui_config().opacity == 1.0
    monkeypatch.setenv("EQ_DASH_OPACITY", "不是数")
    assert theme.load_ui_config().opacity == theme.DEFAULT_OPACITY


def test_config_rejects_bad_primary(_clean_env, monkeypatch, warm_img):
    monkeypatch.setenv("EQ_DASH_IMAGE", str(warm_img))
    monkeypatch.setenv("EQ_DASH_PRIMARY", "红色")
    assert theme.load_ui_config().primary is None
    monkeypatch.setenv("EQ_DASH_PRIMARY", "#a86034")
    assert theme.load_ui_config().primary == "#a86034"


def test_save_image_to_env(tmp_db, warm_img):
    env = theme.save_image_to_env(warm_img)
    text = env.read_text(encoding="utf-8")
    assert f"EQ_DASH_IMAGE={warm_img}" in text
    # 再存一次不重复
    theme.save_image_to_env(warm_img)
    assert env.read_text(encoding="utf-8").count("EQ_DASH_IMAGE=") == 1


def test_save_image_rejects_missing(tmp_db):
    with pytest.raises(FileNotFoundError):
        theme.save_image_to_env(r"Z:\不存在\x.jpg")


# ====================== 图片处理 ======================

def test_image_data_uri_and_downscale(tmp_path):
    big = Image.new("RGB", (4000, 2400), (200, 180, 160))
    p = tmp_path / "big.jpg"
    big.save(p, quality=95)
    uri = theme.image_data_uri(p, max_w=800)
    assert uri and uri.startswith("data:image/jpeg;base64,")
    # 缩放后的 base64 应远小于原图（4000px → 800px）
    assert len(uri) < p.stat().st_size


def test_image_data_uri_missing_file(tmp_path):
    assert theme.image_data_uri(tmp_path / "nope.jpg") is None


def test_image_cache_invalidates_on_mtime(tmp_path):
    p = tmp_path / "x.jpg"
    Image.new("RGB", (50, 50), (255, 0, 0)).save(p)
    a = theme.image_data_uri(p)
    import os
    import time
    Image.new("RGB", (50, 50), (0, 0, 255)).save(p)
    os.utime(p, (time.time() + 10, time.time() + 10))   # 保证 mtime 变化
    b = theme.image_data_uri(p)
    assert a != b, "文件内容变了，缓存必须失效"


# ====================== 取色 ======================

def test_palette_from_warm_image(warm_img):
    pal = theme.extract_palette(warm_img)
    assert pal["is_light"] is True
    for k in ("dominant", "accent", "accent_text", "overlay"):
        assert pal[k].startswith("#") and len(pal[k]) == 7
    # accent 应偏向那块高饱和色（红棕），而不是灰底
    r = int(pal["accent"][1:3], 16)
    b = int(pal["accent"][5:7], 16)
    assert r > b, f"暖图的强调色应偏暖：{pal['accent']}"


def test_palette_dark_image_flips_text_colors(dark_img):
    pal = theme.extract_palette(dark_img)
    assert pal["is_light"] is False
    # 暗图的遮罩应偏黑（亮度低）
    overlay = pal["overlay"]
    lum = sum(int(overlay[i:i + 2], 16) for i in (1, 3, 5)) / 3
    assert lum < 128


def test_palette_missing_file_falls_back(tmp_path):
    pal = theme.extract_palette(tmp_path / "nope.jpg")
    assert pal["dominant"].startswith("#")     # 兜底色，不抛异常


# ====================== CSS ======================

def test_build_css_contains_key_pieces(_clean_env, monkeypatch, warm_img):
    monkeypatch.setenv("EQ_DASH_IMAGE", str(warm_img))
    monkeypatch.setenv("EQ_DASH_OPACITY", "0.75")
    css = theme.build_css(theme.load_ui_config())
    assert "data:image/jpeg;base64," in css
    assert "stAppViewContainer" in css and "stSidebar" in css
    assert "0.75" in css                      # 遮罩不透明度进了 CSS
    assert "stMetric" in css                  # 毛玻璃卡片


def test_build_css_manual_primary_overrides(_clean_env, monkeypatch, warm_img):
    monkeypatch.setenv("EQ_DASH_IMAGE", str(warm_img))
    monkeypatch.setenv("EQ_DASH_PRIMARY", "#123456")
    assert "#123456" in theme.build_css(theme.load_ui_config())


def test_build_css_empty_without_image(_clean_env):
    assert theme.build_css(theme.load_ui_config()) == ""


# ================ Streamlit 原生主题参数 ================

def _flags_dict(flags: list[str]) -> dict[str, str]:
    # strict=True 顺带断言参数是成对的（漏了值会当场炸，而不是静默丢一项）
    return dict(zip(flags[::2], flags[1::2], strict=True))


def test_theme_flags_empty_without_image(_clean_env):
    assert theme.streamlit_theme_flags(theme.load_ui_config()) == []


def test_theme_flags_light_image(_clean_env, monkeypatch, warm_img):
    monkeypatch.setenv("EQ_DASH_IMAGE", str(warm_img))
    d = _flags_dict(theme.streamlit_theme_flags(theme.load_ui_config()))
    assert d["--theme.base"] == "light"
    # 亮色底必须配深色字，否则就是当初那个"深底深字"bug 的镜像
    lum = sum(int(d["--theme.textColor"][i:i + 2], 16) for i in (1, 3, 5)) / 3
    assert lum < 128
    for k in ("--theme.primaryColor", "--theme.backgroundColor",
              "--theme.secondaryBackgroundColor"):
        assert d[k].startswith("#") and len(d[k]) == 7


def test_theme_flags_dark_image_flips_text(_clean_env, monkeypatch, dark_img):
    monkeypatch.setenv("EQ_DASH_IMAGE", str(dark_img))
    d = _flags_dict(theme.streamlit_theme_flags(theme.load_ui_config()))
    assert d["--theme.base"] == "dark"
    lum = sum(int(d["--theme.textColor"][i:i + 2], 16) for i in (1, 3, 5)) / 3
    assert lum > 128, "暗色底必须配浅色字（指标数值看不见的那个 bug）"


@pytest.mark.parametrize("fixture_name", ["warm_img", "dark_img"])
def test_theme_flags_controls_stand_out_from_background(
    _clean_env, monkeypatch, request, fixture_name
):
    """控件背景要和正文背景有区分，否则表格/下拉框糊在背景里。"""
    monkeypatch.setenv("EQ_DASH_IMAGE", str(request.getfixturevalue(fixture_name)))
    d = _flags_dict(theme.streamlit_theme_flags(theme.load_ui_config()))
    bg, second = d["--theme.backgroundColor"], d["--theme.secondaryBackgroundColor"]
    assert bg != second
    diff = max(abs(int(bg[i:i + 2], 16) - int(second[i:i + 2], 16)) for i in (1, 3, 5))
    assert diff >= 3, f"控件背景与正文背景差异太小：{bg} vs {second}"


def test_theme_flags_manual_primary_wins(_clean_env, monkeypatch, warm_img):
    monkeypatch.setenv("EQ_DASH_IMAGE", str(warm_img))
    monkeypatch.setenv("EQ_DASH_PRIMARY", "#123456")
    d = _flags_dict(theme.streamlit_theme_flags(theme.load_ui_config()))
    assert d["--theme.primaryColor"] == "#123456"


def test_theme_flags_never_raise(_clean_env, monkeypatch, warm_img):
    """取色炸了只能退回默认主题，不能让 eq dash 起不来。"""
    monkeypatch.setenv("EQ_DASH_IMAGE", str(warm_img))
    monkeypatch.setattr(theme, "extract_palette",
                        lambda p: (_ for _ in ()).throw(RuntimeError("boom")))
    assert theme.streamlit_theme_flags(theme.load_ui_config()) == []


def test_runner_appends_theme_flags(_clean_env, monkeypatch, warm_img):
    """runner 真把参数拼进 streamlit 命令行了（只有这里能验证端到端）。"""
    monkeypatch.setenv("EQ_DASH_IMAGE", str(warm_img))
    from eq.web import runner

    seen: dict = {}

    class _R:
        returncode = 0

    monkeypatch.setattr(runner.subprocess, "run",
                        lambda cmd, *a, **k: (seen.update(cmd=cmd), _R())[1])
    assert runner.run_dashboard(port=9999) == 0
    assert "--theme.base" in seen["cmd"]
    assert "--server.port" in seen["cmd"] and "9999" in seen["cmd"]


# ====================== apply（假 st） ======================

class _FakeSidebar:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeSt:
    def __init__(self):
        self.markdowns: list[str] = []
        self.sidebar = _FakeSidebar()

    def markdown(self, body, unsafe_allow_html=False):
        self.markdowns.append(body)


def test_apply_injects_css_and_mascot(_clean_env, monkeypatch, warm_img):
    monkeypatch.setenv("EQ_DASH_IMAGE", str(warm_img))
    fake = _FakeSt()
    # sidebar 内的 markdown 也走 fake.markdown（with fake.sidebar 后仍调 st.markdown）
    monkeypatch.setattr(theme, "load_ui_config", theme.load_ui_config)
    info = theme.apply(fake)
    assert info["enabled"] is True
    assert any("<style>" in m for m in fake.markdowns), "应注入 CSS"


def test_apply_disabled_without_config(_clean_env):
    fake = _FakeSt()
    assert theme.apply(fake)["enabled"] is False
    assert fake.markdowns == []


def test_apply_never_raises(monkeypatch, _clean_env, warm_img):
    """主题坏了不能拖垮仪表盘。"""
    monkeypatch.setenv("EQ_DASH_IMAGE", str(warm_img))
    monkeypatch.setattr(theme, "build_css",
                        lambda cfg: (_ for _ in ()).throw(RuntimeError("boom")))
    info = theme.apply(_FakeSt())
    assert info["enabled"] is False
