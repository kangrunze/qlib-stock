# -*- coding: utf-8 -*-
"""
基于 SHAP 的模型可解释性分析 — explain.py

Phase 4 模型能力恢复。
对接 Qlib LGBModel 训练产物，使用 SHAP (SHapley Additive exPlanations)
分解每个特征的边际贡献，帮助理解模型决策逻辑。

核心功能：
  1. 全局特征重要性（SHAP summary plot）
  2. 单样本解释（waterfall plot）
  3. 特征交互效应分析

使用场景：
  - 第 6.2 节：验证新接入的基本面因子是否真的被模型使用
  - 第 10.1 节：判断收益来源是选股能力还是风格暴露

Usage:
    from research.explain import compute_shap_values, generate_shap_report
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from qlib_pipeline.numpy_compat import *  # noqa

logger = logging.getLogger(__name__)


def compute_shap_values(
    model,
    dataset,
    segment: str = "test",
    max_samples: int = 2000,
    background_samples: int = 200,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """计算 SHAP 值，返回特征矩阵、SHAP 值和特征名列表。

    实现策略：
      - Qlib LGBModel 内部封装了 LightGBM Booster，通过 model.model 访问
      - 使用 TreeExplainer（针对树模型的高效实现）
      - 对大样本集进行采样以控制计算时间

    Args:
        model: 已训练好的 Qlib LGBModel 实例
        dataset: Qlib DatasetH 实例
        segment: 数据集段 ("train" | "valid" | "test")
        max_samples: SHAP 计算的最大样本数（采样以避免 OOM）
        background_samples: 背景样本数（用于基线期望值）

    Returns:
        (feature_df, shap_values, feature_names)
            - feature_df: 特征值 DataFrame
            - shap_values: SHAP 值数组 (n_samples, n_features)
            - feature_names: 特征名列表
    """
    try:
        import shap
    except ImportError:
        logger.error("SHAP 未安装，请运行: pip install shap")
        return pd.DataFrame(), np.array([]), []

    # 提取特征数据
    data = dataset.prepare(segment, col_set=["feature"])
    feature_df = data["feature"]

    # 对大样本集采样
    if len(feature_df) > max_samples:
        logger.info("SHAP 计算: 采样 %d/%d 个样本", max_samples, len(feature_df))
        feature_df = feature_df.sample(n=max_samples, random_state=42)

    feature_names = list(feature_df.columns)
    X = feature_df.values

    # 提取 LightGBM Booster
    if hasattr(model, "model") and hasattr(model.model, "booster_"):
        booster = model.model.booster_
    elif hasattr(model, "model"):
        booster = model.model
    else:
        logger.error("无法从模型中提取 LightGBM Booster")
        return feature_df, np.array([]), feature_names

    # 背景数据（用于计算基线期望值）
    if background_samples > 0 and len(feature_df) > background_samples:
        background = feature_df.sample(n=background_samples, random_state=42).values
    else:
        background = X[:min(100, len(X))]

    # 计算 SHAP
    logger.info("SHAP 计算: %d 样本 × %d 特征 ...", len(X), len(feature_names))
    explainer = shap.TreeExplainer(booster, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(X)

    # shap_values 可能是 (n_samples, n_features) 或 list of arrays
    if isinstance(shap_values, list):
        shap_values = shap_values[0] if len(shap_values) > 0 else np.array([])

    logger.info("SHAP 计算完成: shape=%s", shap_values.shape)
    return feature_df, shap_values, feature_names


def get_shap_feature_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """从 SHAP 值计算全局特征重要性（按平均绝对 SHAP 排序）。

    Args:
        shap_values: SHAP 值数组 (n_samples, n_features)
        feature_names: 特征名列表

    Returns:
        DataFrame with columns: feature, mean_abs_shap, std_shap, rank
    """
    if len(shap_values) == 0 or len(feature_names) == 0:
        return pd.DataFrame()

    mean_abs = np.abs(shap_values).mean(axis=0)
    std_shap = shap_values.std(axis=0)

    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
        "std_shap": std_shap,
    })
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    df["cumulative_importance"] = df["mean_abs_shap"].cumsum() / df["mean_abs_shap"].sum()

    return df


def explain_single_prediction(
    shap_values: np.ndarray,
    feature_df: pd.DataFrame,
    feature_names: List[str],
    sample_idx: int = 0,
) -> pd.DataFrame:
    """解释单个样本的预测：每个特征的 SHAP 贡献。

    Args:
        shap_values: SHAP 值数组
        feature_df: 特征值 DataFrame
        feature_names: 特征名列表
        sample_idx: 样本索引

    Returns:
        DataFrame with columns: feature, feature_value, shap_contribution
    """
    if len(shap_values) == 0:
        return pd.DataFrame()

    idx = min(sample_idx, len(shap_values) - 1)
    sample_shap = shap_values[idx]
    sample_features = feature_df.iloc[idx]

    df = pd.DataFrame({
        "feature": feature_names,
        "feature_value": sample_features.values,
        "shap_contribution": sample_shap,
    })
    df = df.sort_values("shap_contribution", key=abs, ascending=False)

    return df


def generate_shap_report(
    model,
    dataset,
    output_dir: str = "output/shap",
    segment: str = "test",
    max_samples: int = 2000,
    top_n_features: int = 20,
) -> Dict:
    """生成完整的 SHAP 可解释性报告。

    产出：
      1. shap_importance.csv — 全局 SHAP 特征重要性排名
      2. shap_summary.png — SHAP summary plot（前 top_n 特征）
      3. shap_report.json — 结构化报告摘要

    Args:
        model: Qlib LGBModel 实例
        dataset: Qlib DatasetH 实例
        output_dir: 输出目录
        segment: 数据段
        max_samples: 最大样本数
        top_n_features: summary plot 显示的特征数

    Returns:
        dict report summary
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 计算 SHAP 值
    feature_df, shap_values, feature_names = compute_shap_values(
        model, dataset, segment=segment, max_samples=max_samples,
    )

    if len(shap_values) == 0:
        logger.error("SHAP 计算失败，无法生成报告")
        return {"error": "shap computation failed"}

    # 全局特征重要性
    importance_df = get_shap_feature_importance(shap_values, feature_names)
    importance_df.to_csv(output_path / "shap_importance.csv", index=False)

    # 生成 SHAP summary plot
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top_features = importance_df.head(top_n_features)["feature"].tolist()
        top_idx = [feature_names.index(f) for f in top_features if f in feature_names]
        top_shap = shap_values[:, top_idx]
        top_feature_df = feature_df.iloc[:, top_idx]

        fig = plt.figure(figsize=(12, 8))
        shap.summary_plot(top_shap, top_feature_df, feature_names=top_features, show=False)
        plt.tight_layout()
        fig.savefig(str(output_path / "shap_summary.png"), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("SHAP summary plot 已保存: shap_summary.png")
    except Exception as e:
        logger.warning("SHAP 图表生成失败: %s", e)

    # 结构化报告
    n_features = len(feature_names)
    top_10 = importance_df.head(10)["feature"].tolist() if len(importance_df) >= 10 else importance_df["feature"].tolist()

    report = {
        "n_samples": len(feature_df),
        "n_features": n_features,
        "top_10_features": top_10,
        "top_feature": importance_df.iloc[0]["feature"] if len(importance_df) > 0 else None,
        "top_feature_importance": float(importance_df.iloc[0]["mean_abs_shap"]) if len(importance_df) > 0 else 0.0,
        "n_features_90pct_cumulative": int((importance_df["cumulative_importance"] <= 0.9).sum()),
        "feature_importance_mean": float(importance_df["mean_abs_shap"].mean()),
        "feature_importance_std": float(importance_df["mean_abs_shap"].std()),
    }

    # 保存 JSON
    import json
    with open(output_path / "shap_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    logger.info(
        "SHAP 报告完成: %d 特征, 前10特征: %s",
        n_features, ", ".join(top_10[:5]),
    )
    return report