"""自写预处理层（v0.38）。

这层以前是丢给 qlib handler 的字典配置，跑一次要 qlib + 一整套 .bin 数据，
边界行为（全 NaN 列、单股票截面、MAD=0 的常数列）从来没验证过。
接管过来最大的收益就是这些用例能跑。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eq.strategy.factors import preprocess as pp


def _panel(n_days=10, n_stocks=20, n_feat=4, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range("2025-01-01", periods=n_days),
         [f"S{i:02d}" for i in range(n_stocks)]],
        names=["datetime", "instrument"])
    return pd.DataFrame(rng.normal(size=(len(idx), n_feat)),
                        index=idx, columns=[f"f{i}" for i in range(n_feat)])


# ---------- ProcessInf ----------

def test_process_inf_replaces_with_column_mean():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.inf], "b": [1.0, 1.0, 1.0, 1.0]})
    out = pp.process_inf(df)
    assert out.loc[3, "a"] == pytest.approx(2.0), "应换成有限值均值 (1+2+3)/3"
    assert np.isfinite(out.to_numpy()).all()


def test_process_inf_handles_negative_inf():
    df = pd.DataFrame({"a": [10.0, 20.0, -np.inf]})
    assert pp.process_inf(df).loc[2, "a"] == pytest.approx(15.0)


def test_process_inf_all_inf_column_becomes_zero():
    """整列都是 inf 时没有有限值可参考，退成 0 而不是 NaN 或崩溃。"""
    df = pd.DataFrame({"a": [np.inf, np.inf], "b": [1.0, 2.0]})
    out = pp.process_inf(df)
    assert (out["a"] == 0.0).all()


def test_process_inf_preserves_nan():
    """NaN 不归它管（后面 fillna 处理），不能顺手填掉。"""
    df = pd.DataFrame({"a": [1.0, np.nan, np.inf]})
    assert pd.isna(pp.process_inf(df).loc[1, "a"])


def test_process_inf_no_inf_is_identity():
    df = _panel(3, 5, 2)
    pd.testing.assert_frame_equal(pp.process_inf(df), df)


# ---------- RobustZScore ----------

def test_robust_zscore_uses_median_not_mean():
    """一个极端离群值不该把正常值全压到 0 附近——这正是不用均值/标准差的原因。"""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 1000.0]})
    p = pp.Pipeline(clip_outlier=False).fit(df)
    out = p.transform(df)
    normal = out["a"].iloc[:4]
    assert normal.std() > 0.3, f"正常值之间应保留区分度：{normal.tolist()}"


def test_robust_zscore_clips_at_three_sigma():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 1e9]})
    out = pp.Pipeline(clip_outlier=True).fit_transform(df)
    assert out["a"].max() <= 3.0 and out["a"].min() >= -3.0


def test_robust_zscore_no_clip_keeps_extremes():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 1e9]})
    assert pp.Pipeline(clip_outlier=False).fit_transform(df)["a"].max() > 3.0


def test_constant_column_does_not_blow_up():
    """MAD=0 的常数列会让分母为 0——必须变成 0 而不是 inf/NaN。"""
    df = pd.DataFrame({"const": [5.0] * 10, "vary": np.arange(10.0)})
    out = pp.Pipeline().fit_transform(df)
    assert np.isfinite(out.to_numpy()).all()
    assert (out["const"] == 0.0).all(), "常数列归一化后应为 0（fillna 兜底）"


def test_median_is_centered_at_zero():
    df = pd.DataFrame({"a": np.arange(101.0)})
    out = pp.Pipeline(clip_outlier=False).fit_transform(df)
    assert out["a"].median() == pytest.approx(0.0, abs=1e-9)


# ---------- fit 只在训练段：防泄漏 ----------

def test_fit_window_excludes_future():
    """在训练段拟合的统计量，不能被验证段的分布带偏。"""
    train = pd.DataFrame({"a": np.arange(100.0)})
    valid = pd.DataFrame({"a": np.arange(1000.0, 1100.0)})     # 分布完全不同

    p = pp.Pipeline(clip_outlier=False).fit(train)
    med_before = float(p.median_["a"])
    p.transform(valid)                                          # 只 transform 不 refit
    assert float(p.median_["a"]) == med_before, "transform 不许改动已拟合的统计量"
    # 验证段整体远高于训练段，归一化后应当整体为正且很大
    assert p.transform(valid)["a"].min() > 3.0


def test_fit_transform_on_full_data_would_leak():
    """反证：全样本 fit 会把验证段的信息吸进统计量，值和只 fit 训练段不同。"""
    train = pd.DataFrame({"a": np.arange(100.0)})
    full = pd.DataFrame({"a": np.concatenate([np.arange(100.0), np.arange(1000.0, 1100.0)])})
    only_train = pp.Pipeline().fit(train).median_["a"]
    leaked = pp.Pipeline().fit(full).median_["a"]
    assert only_train != leaked, "构造前提：两种拟合方式必须给出不同统计量"


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="未 fit"):
        pp.Pipeline().transform(pd.DataFrame({"a": [1.0]}))


def test_transform_rejects_missing_columns():
    """训练和推理的特征集必须一致——少列要当场报错，不能悄悄算。"""
    p = pp.Pipeline().fit(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}))
    with pytest.raises(ValueError, match="特征列缺失"):
        p.transform(pd.DataFrame({"a": [1.0]}))


def test_transform_tolerates_extra_columns():
    """多出来的列忽略即可（只取训练时见过的列，且顺序对齐）。"""
    p = pp.Pipeline().fit(pd.DataFrame({"a": [1.0, 2.0]}))
    out = p.transform(pd.DataFrame({"a": [1.0, 2.0], "zzz": [9.0, 9.0]}))
    assert list(out.columns) == ["a"]


# ---------- 截面归一化 ----------

def test_cs_rank_norm_is_per_day():
    """构造：每天整体抬高一个大常数。截面归一化后各天分布必须一致。"""
    df = _panel(n_days=6, n_stocks=30, n_feat=1)
    day_no = df.index.get_level_values("datetime").factorize()[0]
    s = df["f0"] + day_no * 1000.0                   # 大盘漂移
    out = pp.cs_rank_norm(s)
    per_day_mean = out.groupby(out.index.get_level_values("datetime")).mean()
    assert per_day_mean.abs().max() < 1e-9, (
        "每日均值都该严格为 0。qlib 的 rank(pct=True)-0.5 会残留 +1/(2n) 的偏移，"
        "本项目改用 (rank-(n+1)/2)/n 精确居中")


def test_cs_rank_norm_unit_variance():
    s = _panel(n_days=8, n_stocks=200, n_feat=1)["f0"]
    out = pp.cs_rank_norm(s)
    assert out.std() == pytest.approx(1.0, abs=0.05), "rank 归一化后应接近单位方差"


def test_cs_rank_norm_is_outlier_immune():
    """一个天文数字只影响它所在那天的名次，其他日子分毫不动。

    这正是 rank 归一化的价值：z-score 下一个离群值会把当天所有票的取值
    全部改写，rank 下只有排在它上面的那些名次挪一位。
    """
    s = _panel(n_days=3, n_stocks=20, n_feat=1)["f0"]
    a = pp.cs_rank_norm(s)
    s2 = s.copy()
    s2.iloc[0] = 1e15                       # 改第一天的第一只
    b = pp.cs_rank_norm(s2)

    days = a.index.get_level_values("datetime")
    d0 = days[0]
    other = days != d0
    assert (a[other] == b[other]).all(), "其他日子不该受任何影响"
    # 当天：被改的那只跳到最高，其余票的名次最多挪一位
    same_day = (a[~other] - b[~other]).abs()
    assert same_day.iloc[1:].max() < 0.4, "同日其他票只应挪一个名次"


def test_cs_zscore_keeps_magnitude_info():
    """z-score 保留幅度：涨幅差 10 倍的两只票，归一化后仍差得开。"""
    idx = pd.MultiIndex.from_product(
        [[pd.Timestamp("2025-01-02")], ["A", "B", "C"]],
        names=["datetime", "instrument"])
    s = pd.Series([0.0, 1.0, 10.0], index=idx)
    z = pp.cs_zscore_norm(s)
    r = pp.cs_rank_norm(s)
    assert (z.iloc[2] - z.iloc[1]) > (z.iloc[1] - z.iloc[0]), "z-score 应体现 C 远高于 B"
    assert (r.iloc[2] - r.iloc[1]) == pytest.approx(r.iloc[1] - r.iloc[0]), \
        "rank 只看次序，间距相等"


def test_cs_norm_single_stock_day_is_safe():
    """某天只有一只票时截面标准差为 0，不能产生 inf。"""
    idx = pd.MultiIndex.from_product([[pd.Timestamp("2025-01-02")], ["A"]],
                                     names=["datetime", "instrument"])
    s = pd.Series([1.0], index=idx)
    assert not np.isinf(pp.cs_zscore_norm(s).to_numpy()).any()
    assert np.isfinite(pp.cs_rank_norm(s).to_numpy()).all()


def test_cs_norm_without_date_index_degrades():
    """没有日期索引时退化成全样本，不能报错。"""
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert len(pp.cs_rank_norm(s)) == 4
    assert len(pp.cs_zscore_norm(s)) == 4


# ---------- 标签 ----------

def test_dropna_label_drops_only_label_nan():
    idx = pd.MultiIndex.from_product(
        [[pd.Timestamp("2025-01-02")], ["A", "B", "C"]],
        names=["datetime", "instrument"])
    x = pd.DataFrame({"f": [1.0, np.nan, 3.0]}, index=idx)
    y = pd.Series([0.1, 0.2, np.nan], index=idx)
    xs, ys = pp.dropna_label(x, y)
    assert len(ys) == 2, "只丢标签缺失的那一行"
    assert pd.isna(xs["f"].iloc[1]), "特征缺失保留，交给 fillna"


def test_normalize_label_methods():
    s = _panel(4, 20, 1)["f0"]
    assert pp.normalize_label(s, "rank").std() == pytest.approx(1.0, abs=0.1)
    assert pp.normalize_label(s, "none").equals(s)
    with pytest.raises(ValueError, match="未知标签归一化方式"):
        pp.normalize_label(s, "魔法")


# ---------- 全链路 ----------

def test_pipeline_output_is_finite_and_bounded():
    df = _panel(20, 30, 6)
    df.iloc[0, 0] = np.inf
    df.iloc[1, 1] = np.nan
    df.iloc[2, 2] = 1e12
    df["dead"] = 0.0                       # 常数列
    out = pp.default_pipeline().fit_transform(df)
    assert np.isfinite(out.to_numpy()).all(), "输出必须全是有限值"
    assert out.to_numpy().max() <= 3.0 and out.to_numpy().min() >= -3.0


def test_pipeline_preserves_index():
    df = _panel(5, 10, 3)
    out = pp.default_pipeline().fit_transform(df)
    pd.testing.assert_index_equal(out.index, df.index)


def test_compare_with_qlib_reports_absence_gracefully():
    """没装 qlib 时返回 error 字段，不抛异常。"""
    out = pp.compare_with_qlib(_panel(3, 5, 2))
    assert "error" in out or "max_abs_diff" in out
