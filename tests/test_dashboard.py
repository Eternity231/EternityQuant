"""看板端到端渲染（v0.33）。

价值在于：dashboard.py 是一长串 ``elif page == ...``，某一页里的拼写错误
（未定义变量、改错的 API 名）只有在**选中那一页**时才会炸。之前 CLI 全绿、
测试全过，网页某页照样白屏——这套用例把每一页都真跑一遍。

用 Streamlit 官方的 ``AppTest``：进程内渲染，不起服务器、不开浏览器。
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from eq.web import runner  # noqa: E402

PAGES = ["概览", "晨报", "持仓", "自选", "选股", "回测", "监控规则",
         "ML 模型", "下载管理", "深度研究"]


@pytest.fixture
def app(tmp_db, monkeypatch, tmp_path):
    """空库 + 断开主题配置的干净看板。

    断 DEFAULT_ENV 是因为用户真配了 EQ_DASH_IMAGE 时，主题会去读那张图，
    渲染结果就依赖本机文件了；测试不该有这种外部依赖。
    """
    from eq.core import python_dotenv_loader as loader

    monkeypatch.setattr(loader, "DEFAULT_ENV", tmp_path / "nonexistent.env")
    monkeypatch.delenv("EQ_DASH_IMAGE", raising=False)
    return AppTest.from_file(str(runner._ENTRY), default_timeout=60)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(app, page):
    """每一页都能渲染出来且不抛异常。"""
    app.run()
    assert not app.exception, f"首屏就炸了：{app.exception}"
    app.selectbox[0].select(page).run()
    assert not app.exception, f"{page} 页渲染异常：{app.exception}"
    # 页面标题会以 header 形式出现，至少要有内容渲染出来
    assert app.markdown or app.header or app.dataframe or app.info


def test_backtest_page_lists_all_strategies(app):
    """回测页的策略下拉必须来自共享注册表，不是手抄的那 4 个。"""
    from eq.strategy.registry import list_strategies

    app.run()
    app.selectbox[0].select("回测").run()
    opts = [o for sb in app.selectbox for o in sb.options]
    for name in list_strategies():
        assert name in opts, f"回测页缺策略 {name}"


def test_briefing_page_shows_empty_scoreboard(app):
    """空库时纸面战绩给提示而不是报错。"""
    app.run()
    app.selectbox[0].select("晨报").run()
    assert not app.exception
    assert any("尚无已结算记录" in i.value for i in app.info)
