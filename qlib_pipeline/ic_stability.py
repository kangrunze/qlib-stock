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


def compute_icir(ic_series: pd.Series) -> Dict[str, float]:
    """
    Compute IC Information Ratio and related metrics.

    ICIR = mean(IC) / std(IC)

    Interpretation:
      ICIR > 0.5 : Acceptable
      ICIR > 1.0 : Excellent

    Returns:
        Dict with ic_mean, ic_std, icir, ic_positive_ratio, ic_t_stat
    """
    ic = ic_series.dropna()
    if len(ic) < 5:
        return {"ic_mean": np.nan, "ic_std": np.nan, "icir": np.nan,
                "ic_positive_ratio": np.nan, "ic_t_stat": np.nan}

    ic_mean = float(ic.mean())
    ic_std = float(ic.std())
    icir = ic_mean / ic_std if ic_std > 0 else np.nan
    ic_positive_ratio = float((ic > 0).mean())
    ic_t_stat = ic_mean / (ic_std / np.sqrt(len(ic))) if ic_std > 0 else np.nan

    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "icir": float(icir),
        "ic_positive_ratio": ic_positive_ratio,
        "ic_t_stat": float(ic_t_stat),
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
    return icir