# -*- coding: utf-8 -*-
"""
IC 稳定性增强 — ic_stability.py

Phase 1 稳健性基础设施。
- ICIR (Information Coefficient Information Ratio)
- IC 衰减分析 (Decay)
- IC 分层分析 (Stratified IC by market cap / industry)
- IC 自相关分析

Usage:
    from qlib_pipeline.ic_stability import compute_icir, ic_decay, stratified_ic
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from qlib_pipeline.numpy_compat import *  # noqa

logger = logging.getLogger(__name__)


def rank_ic(predictions: pd.Series, labels: pd.Series) -> float:
    """Compute rank IC (Spearman correlation)."""
    mask = predictions.notna() & labels.notna()
    if mask.sum() < 10:
        return np.nan
    return float(stats.spearmanr(predictions[mask], labels[mask]).correlation)


def compute_daily_rank_ic(pred: pd.Series, label: pd.Series) -> pd.Series:
    """按日期分组计算逐日 RankIC（Spearman 相关系数）。

    pred/label 均为 MultiIndex(datetime, instrument) 的 Series。
    返回 index 为日期、值为当日 RankIC 的 Series。

    实现说明：Spearman 相关 = Pearson(组内 rank)。先用 groupby.rank() 向量化
    计算组内排名，再用组内标准化 + 点积求 Pearson，避免逐日 apply(lambda) 的
    Python 回调开销，在大样本下有显著加速。
    """
    df = pd.DataFrame({"pred": pred, "label": label}).dropna()
    if df.empty:
        return pd.Series(dtype=float)

    # 过滤每日样本数 < 3 的日期（少于 3 个样本无统计意义）
    counts = df.groupby(level=0).size()
    valid_dates = counts[counts >= 3].index
    df = df[df.index.get_level_values(0).isin(valid_dates)]
    if df.empty:
        return pd.Series(dtype=float)

    # 组内 rank（Spearman = Pearson of ranks）
    ranked = df.groupby(level=0).rank()

    # 组内标准化后点积 = Pearson 相关
    grouped = ranked.groupby(level=0)
    pred_z = grouped["pred"].transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0)
    label_z = grouped["label"].transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0)

    # 组内均值即为 Pearson 相关系数
    daily_ic = (pred_z * label_z).groupby(level=0).mean()
    return daily_ic.dropna()


def get_ic_series(model, dataset, segment: str = "test") -> pd.Series:
    """在指定段上预测并计算逐日 RankIC 序列。

    本函数封装了"预测 → 计算 IC"这一核心逻辑，供 run.py 中
    cmd_drift / cmd_ic_stability / cmd_regime 以及 evaluate_fold() 统一调用，
    避免多处重复实现导致不一致。

    Args:
        model: 已训练好的 Qlib 模型
        dataset: Qlib DatasetH 实例
        segment: 数据段名称，默认 "test"。
            超参搜索（Optuna）应传 "valid" 以避免验证集泄露。

    Returns:
        pd.Series，index 为日期，值为当日 RankIC（Spearman）
    """
    data = dataset.prepare(segment, col_set=["feature", "label"])
    preds = model.predict(dataset, segment=segment)
    labels = data["label"]
    return compute_daily_rank_ic(
        preds,
        labels.iloc[:, 0] if labels.ndim > 1 else labels,
    )


def evaluate_fold(model, dataset, label_horizon: int = 20) -> dict:
    """在测试集上评估模型表现，返回 IC 汇总指标 + Newey-West 显著性检验。

    内部调用 get_ic_series() 获取逐日 IC 序列后计算均值/标准差/ICIR，
    并通过 research.significance.newey_west_test() 进行 Newey-West 调整后的
    显著性检验（处理标签重叠导致的自相关）。

    用于 rolling.py / tscv.py 每折训练后的评估环节。

    Args:
        model: 已训练好的 Qlib 模型
        dataset: Qlib DatasetH 实例
        label_horizon: 标签前瞻天数，用于 Newey-West 滞后阶数

    Returns:
        dict with ic_mean, ic_std, icir, ic_positive_ratio,
             nw_t_stat, nw_p_value, nw_significant, n_samples, nw_lags
    """
    try:
        ic_series = get_ic_series(model, dataset)
        if len(ic_series) == 0:
            return {
                "ic_mean": None, "ic_std": None, "icir": None,
                "ic_positive_ratio": None,
                "nw_t_stat": None, "nw_p_value": None, "nw_significant": False,
                "n_samples": 0, "nw_lags": 0,
            }

        ic_mean = float(ic_series.mean())
        ic_std = float(ic_series.std())
        icir = ic_mean / ic_std if ic_std > 0 else float("nan")
        ic_positive_ratio = float((ic_series > 0).mean())

        # Newey-West 显著性检验（第 8.1 节）
        try:
            from research.significance import newey_west_test
            nw_result = newey_west_test(ic_series, max_lags=label_horizon)
            nw_t_stat = nw_result.get("t_nw")
            nw_p_value = nw_result.get("p_value_nw")
            nw_significant = nw_result.get("significant", False)
            n_samples = nw_result.get("n_samples", len(ic_series))
            nw_lags = nw_result.get("max_lags", 0)
        except ImportError:
            nw_t_stat = ic_mean / (ic_std / np.sqrt(len(ic_series))) if ic_std > 0 else float("nan")
            nw_p_value = None
            nw_significant = False
            n_samples = len(ic_series)
            nw_lags = 0

        return {
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "icir": icir,
            "ic_positive_ratio": ic_positive_ratio,
            "nw_t_stat": nw_t_stat,
            "nw_p_value": nw_p_value,
            "nw_significant": nw_significant,
            "n_samples": n_samples,
            "nw_lags": nw_lags,
        }
    except Exception as e:
        logger.warning("折内评估失败: %s", e)
        return {
            "ic_mean": None, "ic_std": None, "icir": None,
            "ic_positive_ratio": None,
            "nw_t_stat": None, "nw_p_value": None, "nw_significant": False,
            "n_samples": 0, "nw_lags": 0,
        }


def get_feature_importance(model, dataset) -> dict:
    """从训练好的 Qlib LGBModel 中提取特征重要性。

    用于 rolling.py / tscv.py 每折训练后导出特征重要性，
    为后续 feature_stability() 概念漂移检测提供跨折特征重要性列表。

    Args:
        model: 已训练好的 Qlib LGBModel 实例
        dataset: Qlib DatasetH 实例

    Returns:
        dict of {feature_name: importance_score}，提取失败时返回空 dict
    """
    try:
        # 获取特征名称
        test_data = dataset.prepare("test", col_set=["feature"])
        feature_names = list(test_data["feature"].columns)

        # 提取 LightGBM 底层模型的特征重要性
        if hasattr(model, "model") and hasattr(model.model, "feature_importance"):
            importances = model.model.feature_importance(importance_type="gain")
            # 确保长度匹配
            if len(importances) == len(feature_names):
                return dict(zip(feature_names, importances.tolist()))
            elif len(importances) < len(feature_names):
                # 部分特征未被使用，补 0
                result = dict(zip(feature_names, [0.0] * len(feature_names)))
                for i, imp in enumerate(importances):
                    if i < len(feature_names):
                        result[feature_names[i]] = float(imp)
                return result
            else:
                logger.warning("特征重要性数量(%d)与特征数量(%d)不匹配，截断处理", len(importances), len(feature_names))
                return dict(zip(feature_names, importances[:len(feature_names)].tolist()))
        else:
            logger.warning("模型不支持 feature_importance 接口，无法提取特征重要性")
            return {}
    except Exception as e:
        logger.warning("提取特征重要性失败: %s", e)
        return {}


def compute_icir(ic_series: pd.Series, max_lags: Optional[int] = None) -> Dict[str, float]:
    """
    Compute IC Information Ratio and related metrics, including Newey-West adjusted significance.

    ICIR = mean(IC) / std(IC)

    Interpretation:
      ICIR > 0.5 : Acceptable
      ICIR > 1.0 : Excellent

    Args:
        ic_series: 逐日 IC 序列
        max_lags: Newey-West 最大滞后阶数（默认 = 标签周期长度）

    Returns:
        Dict with ic_mean, ic_std, icir, ic_positive_ratio,
             naive_t_stat, nw_t_stat, nw_p_value, nw_significant
    """
    ic = ic_series.dropna()
    if len(ic) < 5:
        return {
            "ic_mean": np.nan, "ic_std": np.nan, "icir": np.nan,
            "ic_positive_ratio": np.nan,
            "naive_t_stat": np.nan, "nw_t_stat": np.nan,
            "nw_p_value": np.nan, "nw_significant": False,
        }

    ic_mean = float(ic.mean())
    ic_std = float(ic.std())
    icir = ic_mean / ic_std if ic_std > 0 else np.nan
    ic_positive_ratio = float((ic > 0).mean())
    naive_t_stat = ic_mean / (ic_std / np.sqrt(len(ic))) if ic_std > 0 else np.nan

    # Newey-West 显著性检验（第 8.1 节）
    try:
        from research.significance import newey_west_test
        nw_result = newey_west_test(ic, max_lags=max_lags)
        nw_t_stat = nw_result.get("t_nw")
        nw_p_value = nw_result.get("p_value_nw")
        nw_significant = nw_result.get("significant", False)
    except ImportError:
        nw_t_stat = naive_t_stat
        nw_p_value = 2.0 * (1.0 - stats.norm.cdf(abs(naive_t_stat))) if not np.isnan(naive_t_stat) else np.nan
        nw_significant = abs(naive_t_stat) > 2.0

    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "icir": float(icir),
        "ic_positive_ratio": ic_positive_ratio,
        "naive_t_stat": float(naive_t_stat),
        "nw_t_stat": nw_t_stat,
        "nw_p_value": nw_p_value,
        "nw_significant": nw_significant,
    }


def ic_decay(predictions: pd.DataFrame, returns: Dict[int, pd.DataFrame]
             ) -> pd.DataFrame:
    """
    Analyze IC decay across different holding periods.

    Args:
        predictions: model predictions (index=date, columns=stock)
        returns: dict of future returns for different horizons
                 e.g., {5: df_ret_5d, 10: df_ret_10d, 20: df_ret_20d, 60: df_ret_60d}

    Returns:
        DataFrame with columns: horizon, ic_mean, ic_std, icir
    """
    results = []
    for horizon, ret_df in sorted(returns.items()):
        # Align dates and stocks
        common_dates = predictions.index.intersection(ret_df.index)
        common_stocks = predictions.columns.intersection(ret_df.columns)
        if len(common_dates) < 5 or len(common_stocks) < 5:
            continue

        pred = predictions.loc[common_dates, common_stocks]
        ret = ret_df.loc[common_dates, common_stocks]

        # Compute daily rank IC
        ic_vals = []
        for d in common_dates:
            pred_d = pred.loc[d].dropna()
            ret_d = ret.loc[d].dropna()
            common = pred_d.index.intersection(ret_d.index)
            if len(common) > 10:
                ic = rank_ic(pred_d.loc[common], ret_d.loc[common])
                ic_vals.append(ic)

        if ic_vals:
            ic_series = pd.Series(ic_vals)
            metrics = compute_icir(ic_series)
            results.append({
                "horizon": horizon,
                **metrics,
                "n_observations": len(ic_vals),
            })

    df = pd.DataFrame(results)
    if not df.empty:
        logger.info("IC 衰减分析完成: %d horizons", len(df))
    return df


def stratified_ic(predictions: pd.Series, labels: pd.Series,
                  groups: pd.Series, n_bins: int = 5) -> pd.DataFrame:
    """
    Stratified IC analysis by group (e.g., market cap quantile).

    Args:
        predictions: model predictions
        labels: actual returns
        groups: grouping variable (e.g., market cap, industry)
        n_bins: number of quantile bins

    Returns:
        DataFrame with IC per group
    """
    # Align data
    common = predictions.index.intersection(labels.index).intersection(groups.index)
    pred = predictions.loc[common].dropna()
    lab = labels.loc[common].dropna()
    grp = groups.loc[common].dropna()
    common = pred.index.intersection(lab.index).intersection(grp.index)
    pred, lab, grp = pred.loc[common], lab.loc[common], grp.loc[common]

    if len(common) < 20:
        return pd.DataFrame()

    if grp.dtype == 'object':
        # Categorical grouping
        results = []
        for g in grp.unique():
            mask = grp == g
            if mask.sum() > 10:
                ic = rank_ic(pred[mask], lab[mask])
                results.append({"group": str(g), "ic": ic, "n": int(mask.sum())})
        return pd.DataFrame(results).sort_values("ic", ascending=False)
    else:
        # Numeric grouping: quantile bins
        try:
            bins = pd.qcut(grp, n_bins, duplicates='drop')
            results = []
            for b in bins.cat.categories:
                mask = bins == b
                if mask.sum() > 10:
                    ic = rank_ic(pred[mask], lab[mask])
                    results.append({"group": str(b), "ic": ic, "n": int(mask.sum())})
            return pd.DataFrame(results)
        except Exception:
            return pd.DataFrame()


def ic_autocorrelation(ic_series: pd.Series, max_lag: int = 20) -> pd.DataFrame:
    """
    Compute IC autocorrelation to assess IC persistence.

    Returns:
        DataFrame with columns: lag, autocorrelation, significance
    """
    ic = ic_series.dropna()
    if len(ic) < max_lag * 2:
        return pd.DataFrame()

    results = []
    for lag in range(1, min(max_lag + 1, len(ic) // 2)):
        acf = pd.Series(ic).autocorr(lag=lag)
        results.append({"lag": lag, "autocorrelation": acf})

    return pd.DataFrame(results)


def generate_ic_stability_report(ic_series: pd.Series,
                                 decay_df: Optional[pd.DataFrame] = None,
                                 stratified_df: Optional[pd.DataFrame] = None,
                                 output_dir: str = "output/ic_stability") -> Dict:
    """Generate comprehensive IC stability report."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    icir = compute_icir(ic_series)
    acf_df = ic_autocorrelation(ic_series)

    # Save data
    ic_series.to_csv(output_path / "ic_series.csv", index=True, header=["ic"])
    if decay_df is not None and not decay_df.empty:
        decay_df.to_csv(output_path / "ic_decay.csv", index=False)
    if stratified_df is not None and not stratified_df.empty:
        stratified_df.to_csv(output_path / "ic_stratified.csv", index=False)
    if not acf_df.empty:
        acf_df.to_csv(output_path / "ic_autocorrelation.csv", index=False)

    # Generate Plotly charts
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("IC Time Series", "IC Decay", "IC Autocorrelation"),
            vertical_spacing=0.12,
        )

        # IC time series
        fig.add_trace(go.Scatter(
            x=list(range(len(ic_series))),
            y=ic_series.values,
            mode='lines',
            name='IC',
            line=dict(color='steelblue', width=1),
        ), row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

        # IC decay
        if decay_df is not None and not decay_df.empty:
            fig.add_trace(go.Scatter(
                x=decay_df["horizon"].values,
                y=decay_df["ic_mean"].values,
                mode='lines+markers',
                name='IC Mean',
                line=dict(color='steelblue', width=2),
            ), row=2, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

        # IC autocorrelation
        if not acf_df.empty:
            fig.add_trace(go.Bar(
                x=acf_df["lag"].values,
                y=acf_df["autocorrelation"].values,
                name='ACF',
                marker_color='steelblue',
            ), row=3, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

        fig.update_layout(
            title=f"IC Stability Report (ICIR={icir['icir']:.4f})",
            template='plotly_dark',
            height=1200,
            width=1280,
            showlegend=False,
        )
        fig.write_image(str(output_path / "ic_stability_report.png"), width=1280, height=1200, scale=2)
        logger.info("IC 稳定性图表已保存: ic_stability_report.png")
    except Exception as e:
        logger.warning("IC 图表生成失败: %s", e)

    logger.info("IC 稳定性报告: ICIR=%.4f, IC_mean=%.4f, IC_+ratio=%.2f%%",
                 icir["icir"], icir["ic_mean"], icir["ic_positive_ratio"] * 100)
    if icir.get("nw_significant") is not None:
        if icir["nw_significant"]:
            logger.info("  → Newey-West 显著性检验: |t|=%.2f (p=%.4f) ✓ 通过 (|t| > 2.0)",
                         icir.get("nw_t_stat", 0), icir.get("nw_p_value", 1))
        else:
            logger.warning(
                "  → Newey-West 显著性检验: |t|=%.2f (p=%.4f) ✗ 未通过 (|t| ≤ 2.0)，"
                "结论应降级为'探索性发现'",
                icir.get("nw_t_stat", 0), icir.get("nw_p_value", 1))
    return icir