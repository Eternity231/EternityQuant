"""数据导出（v0.24 新增）。"""

from __future__ import annotations

import pandas as pd
import pytest

from eq.core import exporter, portfolio as pf, watchlist as wl


@pytest.fixture
def seeded(tmp_db, monkeypatch):
    """填点数据；持仓导出会拉行情，这里 stub 掉。"""
    monkeypatch.setattr("eq.data.market.get_snapshots", lambda syms, **k: {s: None for s in syms})
    wl.add("600519.SH", reason="白酒龙头", tags="白酒")
    wl.add("000001.SZ", tags="银行")
    pf.open_position("600519.SH", 100, 1680.0, stop_loss=1600.0)
    pf.trim("600519.SH", 30, 1700.0)
    return tmp_db


def test_datasets_registry():
    assert set(exporter.DATASETS) == {
        "watchlist", "positions", "closed", "trades", "rules", "signals", "backtests"
    }
    for name, (fn, label) in exporter.DATASETS.items():
        assert callable(fn) and label, name


def test_export_csv(seeded, tmp_path):
    out = tmp_path / "exp"
    r = exporter.export(["watchlist", "trades"], out_dir=out, fmt="csv")
    assert r["rows"]["watchlist"] == 2
    assert r["rows"]["trades"] == 2          # buy + trim
    assert (out / "watchlist.csv").exists()
    assert (out / "trades.csv").exists()

    df = pd.read_csv(out / "watchlist.csv", encoding="utf-8-sig")
    assert set(df["symbol"]) == {"600519.SH", "000001.SZ"}
    assert "白酒龙头" in df["reason"].astype(str).tolist()


def test_export_csv_has_bom_for_excel(seeded, tmp_path):
    """不带 BOM 的 UTF-8 CSV 在 Excel 里双击打开中文是乱码。"""
    out = tmp_path / "exp"
    exporter.export(["watchlist"], out_dir=out, fmt="csv")
    assert (out / "watchlist.csv").read_bytes().startswith(b"\xef\xbb\xbf")


def test_export_all_defaults(seeded, tmp_path):
    r = exporter.export(None, out_dir=tmp_path / "all", fmt="csv")
    assert "watchlist" in r["rows"]
    assert "positions" in r["rows"]
    # 没数据的表进 skipped 而不是写出空文件
    assert any("closed" in s for s in r["skipped"])


def test_export_excel(seeded, tmp_path):
    pytest.importorskip("openpyxl")
    out = tmp_path / "xl"
    r = exporter.export(["watchlist", "trades"], out_dir=out, fmt="excel")
    path = out / "eternityquant.xlsx"
    assert path.exists()
    assert r["files"] == [str(path)]
    sheets = pd.read_excel(path, sheet_name=None)
    assert set(sheets) == {"watchlist", "trades"}


def test_export_rejects_unknown_dataset(seeded, tmp_path):
    with pytest.raises(ValueError, match="未知数据集"):
        exporter.export(["不存在"], out_dir=tmp_path)


def test_export_rejects_bad_format(seeded, tmp_path):
    with pytest.raises(ValueError, match="fmt"):
        exporter.export(["watchlist"], out_dir=tmp_path, fmt="pdf")


def test_export_empty_db(tmp_db, tmp_path, monkeypatch):
    monkeypatch.setattr("eq.data.market.get_snapshots", lambda syms, **k: {})
    r = exporter.export(None, out_dir=tmp_path / "empty", fmt="csv")
    assert r["files"] == []
    assert len(r["skipped"]) == len(exporter.DATASETS)
