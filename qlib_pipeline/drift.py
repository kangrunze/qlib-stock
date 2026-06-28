# -*- coding: utf-8 -*-
"""
特征漂移与概念漂移检测 — drift.py

Phase 1 + Phase 3 稳健性基础设施。
- PSI (Population Stability Index) 特征漂移检测
- 概念漂移检测（滚动 IC 趋势 + 断点检测）
- 特征重要性稳定性（Spearman 相关性）

Usage:
    from qlib_pipeline.drift import compute_psi, detect_concept_drift, feature_stability
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


# ===== PSI 特征漂移检测 =====

def _psi_bucket(series: pd.Series, bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Equal-width binning, returns (bin_edges, expected_distribution)."""
    counts, edges = np.histogram(series.dropna(), bins=bins)
    expected = counts / counts.sum()
    expected = np.clip(expected, 1e-6, None)  # Avoid log(0)
    expected = expected / expected.sum()
    return edges, expected


def compute_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """
    Compute Population Stability Index between two feature distributions.

    PSI < 0.1   : No significant drift
    PSI 0.1-0.25: Moderate drift
    PSI > 0.25  : Significant drift

    Args:
        expected: baseline (training) distribution
        actual: current (test) distribution
        bins: number of bins

    Returns:
        PSI value
    """
    edges, expected_ratio = _psi_bucket(expected, bins)
    actual_counts, _ = np.histogram(actual.dropna(), bins=edges)
    actual_ratio = actual_counts / actual_counts.sum()
    actual_ratio = np.clip(actual_ratio, 1e-6, None)
    actual_ratio = actual_ratio / actual_ratio.sum()

    psi = np.sum((actual_ratio - expected_ratio) * np.log(actual_ratio / expected_ratio))
    return float(psi)


def compute_psi_dataframe(train_df: pd.DataFrame, test_df: pd.DataFrame,
                          bins: int = 10) -> pd.DataFrame:
    """
    Compute PSI for all features in a DataFrame.

    Args:
        train_df: training features
        test_df: test features
        bins: number of bins

    Returns:
        DataFrame with columns: feature, psi, drift_level
    """
    results = []
    common_cols = train_df.columns.intersection(test_df.columns)

    for col in common_cols:
        psi = compute_psi(train_df[col], test_df[col], bins)
        if psi < 0.1:
            level = "low"
        elif psi < 0.25:
            level = "moderate"
        else:
            level = "significant"
        results.append({"feature": col, "psi": psi, "drift_level": level})

    df = pd.DataFrame(results).sort_values("psi", ascending=False)
    logger.info("PSI 检测完成: %d 特征, 显著漂移=%d, 中等=%d",
                 len(df),
                 len(df[df["drift_level"] == "significant"]),
                 len(df[df["drift_level"] == "moderate"]))
    return df


# ===== 概念漂移检测 =====

def detect_concept_drift(ic_series: pd.Series, window: int = 60,
                         threshold: float = 0.3) -> pd.DataFrame:
    """
    Detect concept drift via rolling IC analysis.

    Checks for:
      - Trend decline in rolling IC
      - Structural breaks

    Args:
        ic_series: daily/weekly IC values indexed by date
        window: rolling window size
        threshold: IC decline ratio threshold for drift warning

    Returns:
        DataFrame with rolling_ic, warning flags
    """
    rolling_ic = ic_series.rolling(window=window).mean()
    rolling_std = ic_series.rolling(window=window).std()

    # Detect decline: compare first half vs second half of rolling window
    half = window // 2
    trend = rolling_ic.rolling(window=window).apply(
        lambda x: stats.linregress(np.arange(len(x)), x)[0], raw=True
    )

    results = pd.DataFrame({
        "date": ic_series.index,
        "ic": ic_series.values,
        "rolling_ic": rolling_ic.values,
        "rolling_std": rolling_std.values,
        "trend": trend.values,
    })

    # Flag drift warning
    results["drift_warning"] = False
    if len(results) > window:
        baseline = rolling_ic.iloc[window:window*2].mean()
        recent = rolling_ic.iloc[-window:].mean()
        if pd.notna(baseline) and pd.notna(recent) and baseline > 0:
            decline = (baseline - recent) / baseline
            if decline > threshold:
                results.iloc[-window:, results.columns.get_loc("drift_warning")] = True
                logger.warning("概念漂移检测: IC 下降 %.1f%% (基线=%.4f, 当前=%.4f)",
                               decline * 100, baseline, recent)

    return results


# ===== 特征重要性稳定性 =====

def feature_stability(feature_importance_list: List[pd.Series]) -> float:
    """
    Measure feature importance stability across rolling folds.

    Args:
        feature_importance_list: list of feature importance Series, one per fold

    Returns:
        Mean Spearman rank correlation across consecutive folds
    """
    if len(feature_importance_list) < 2:
        return 1.0

    correlations = []
    for i in range(len(feature_importance_list) - 1):
        fi1 = feature_importance_list[i].sort_values(ascending=False)
        fi2 = feature_importance_list[i + 1].sort_values(ascending=False)
        common = fi1.index.intersection(fi2.index)
        if len(common) > 5:
            corr = stats.spearmanr(fi1.loc[common], fi2.loc[common]).correlation
            if pd.notna(corr):
                correlations.append(corr)

    if not correlations:
        return 0.0

    mean_corr = np.mean(correlations)
    logger.info("特征重要性稳定性: Spearman ρ = %.4f (n_folds=%d)", mean_corr, len(correlations))
    return float(mean_corr)


def generate_drift_report(psi_df: pd.DataFrame, drift_df: pd.DataFrame,
                          stability_score: float,
                          output_dir: str = "output/drift") -> Dict:
    """
    Generate comprehensive drift detection report.

    Returns:
        Dict with summary metrics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report = {
        "psi": {
            "n_features": len(psi_df),
            "n_significant_drift": int(len(psi_df[psi_df["drift_level"] == "significant"])),
            "n_moderate_drift": int(len(psi_df[psi_df["drift_level"] == "moderate"])),
            "top_drift_features": psi_df.head(10)["feature"].tolist() if len(psi_df) > 0 else [],
        },
        "concept_drift": {
            "drift_warnings": int(drift_df["drift_warning"].sum()) if len(drift_df) > 0 else 0,
            "recent_rolling_ic": float(drift_df["rolling_ic"].iloc[-1]) if len(drift_df) > 0 else 0.0,
        },
        "feature_stability": {
            "spearman_rho": stability_score,
            "status": "stable" if stability_score > 0.7 else "warning" if stability_score > 0.5 else "critical",
        },
    }

    # Save report
    import json
    with open(output_path / "drift_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    psi_df.to_csv(output_path / "psi_features.csv", index=False)
    drift_df.to_csv(output_path / "concept_drift.csv", index=False)

    logger.info("漂移检测报告已保存到: %s", output_dir)
    return report