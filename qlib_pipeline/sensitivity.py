# -*- coding: utf-8 -*-
"""
超参数敏感性分析 — sensitivity.py

Phase 3 鲁棒性深化。
- 单变量超参数敏感性扫描
- 敏感性热力图

Usage:
    from qlib_pipeline.sensitivity import hyperparameter_sensitivity
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from qlib_pipeline.numpy_compat import *  # noqa

logger = logging.getLogger(__name__)


def _init_qlib_env(config: dict):
    import os
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    import qlib
    from qlib.constant import REG_CN
    provider_uri = os.environ.get("QLIB_PROVIDER_URI") or config.get("qlib", {}).get("provider_uri")
    if not provider_uri:
        raise ValueError("未设置 QLIB_PROVIDER_URI 环境变量，且配置文件中也未指定 qlib.provider_uri")
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    from qlib.config import C
    C.joblib_backend = "threading"
    C.dataset_process_n_worker = 1


def _single_train(config: dict, param_overrides: dict,
                  dataset=None) -> tuple:
    """Train a single model with given params, return (final valid l2, dataset).

    Qlib 环境已由调用方（hyperparameter_sensitivity）通过 _init_qlib_env 初始化，
    本函数不再重复初始化，避免多次 init 导致资源泄漏。

    通过 train.build_task() 构建任务，确保 label 表达式、CSMedianSubtract、
    RankLGBModel（loss=rank）路由等逻辑与主管线完全一致。

    优化：dataset 可复用（handler 配置不变时数据相同），避免每次重新加载 ~10 分钟数据。
    """
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R
    from qlib_pipeline.train import build_task
    import copy as _copy

    # 深拷贝 config 并合并参数覆盖
    merged_config = _copy.deepcopy(config)
    model_kwargs = merged_config.setdefault("qlib_lgb", {}).setdefault("kwargs", {})
    model_kwargs.update(param_overrides)

    # 通过 build_task 构建 task dict，确保与主管线逻辑一致
    task = build_task(merged_config)

    try:
        # 复用 dataset（handler 配置不变时数据相同，避免重复加载 ~10 分钟）
        if dataset is None:
            dataset = init_instance_by_config(task["dataset"])
        model = init_instance_by_config(task["model"])

        with R.start(experiment_name="sensitivity_scan"):
            model.fit(dataset)
            # RankLGBModel 没有 evals_result，改用 IC 指标（越小越好 → 取 -IC）
            if hasattr(model, "model") and hasattr(model.model, "best_iteration"):
                # RankLGBModel 路径：用 valid IC 作为评估指标（返回 -IC，越小越好）
                from qlib_pipeline.ic_stability import get_ic_series
                ic_series = get_ic_series(model, dataset)
                if len(ic_series) > 0:
                    return -float(ic_series.mean()), dataset  # 取负号使"越小越好"与 l2 一致
                logger.warning("RankLGBModel valid IC 为空，标记为 None")
                return None, dataset
            # Qlib LGBModel 路径：提取 valid l2
            if hasattr(model, "evals_result") and "valid" in model.evals_result:
                valid_scores = model.evals_result.get("valid", {}).get("l2", [])
                if valid_scores:
                    return float(min(valid_scores)), dataset
            elif hasattr(model, "evals_result_") and "valid" in model.evals_result_:
                valid_scores = model.evals_result_.get("valid", {}).get("l2", [])
                if valid_scores:
                    return float(min(valid_scores)), dataset
            logger.warning("无法从模型中提取 valid l2 分数，标记为提取失败（None），请检查 Qlib 版本兼容性")
            return None, dataset
    except Exception as e:
        logger.warning("训练失败 (%s): %s", param_overrides, e)
        return None, dataset


def hyperparameter_sensitivity(config: dict, param_name: str,
                               values: List[float],
                               output_dir: str = "output/sensitivity",
                               shared_dataset: Optional[object] = None) -> pd.DataFrame:
    """
    Single-variable hyperparameter sensitivity scan.

    Scans a range of values for one hyperparameter, recording the
    final valid l2 score for each.

    Args:
        config: workflow config dict
        param_name: parameter name to scan (e.g., "learning_rate", "num_leaves")
        values: list of values to try
        output_dir: output directory
        shared_dataset: 可选，复用的 DatasetH 实例；为 None 时新建

    Returns:
        DataFrame with columns: param_value, valid_l2
    """
    if shared_dataset is None:
        _init_qlib_env(config)

    logger.info("超参数敏感性扫描: %s, values=%s", param_name, values)

    # 复用 dataset：第一次创建后供后续所有扫描点使用
    results = []
    for val in values:
        override = {param_name: val}
        score, shared_dataset = _single_train(config, override, dataset=shared_dataset)
        if score is not None:
            results.append({"param_value": val, "valid_l2": score})
            logger.info("  %s=%.4f -> valid_l2=%.6f", param_name, val, score)

    df = pd.DataFrame(results)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path / f"sensitivity_{param_name}.csv", index=False)

    # Generate sensitivity chart
    if not df.empty:
        _generate_sensitivity_chart(param_name, df, output_path)

    return df


def _generate_sensitivity_chart(param_name: str, df: pd.DataFrame,
                                output_path: Path):
    """Generate sensitivity line chart."""
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["param_value"].values,
            y=df["valid_l2"].values,
            mode='lines+markers',
            name='valid_l2',
            line=dict(color='steelblue', width=2),
            marker=dict(size=8),
        ))

        # Mark best value
        best_idx = df["valid_l2"].idxmin()
        fig.add_trace(go.Scatter(
            x=[df.loc[best_idx, "param_value"]],
            y=[df.loc[best_idx, "valid_l2"]],
            mode='markers',
            name='Best',
            marker=dict(color='red', size=12, symbol='star'),
        ))

        fig.update_layout(
            title=f"Hyperparameter Sensitivity: {param_name}",
            xaxis_title=param_name,
            yaxis_title="valid_l2",
            template='plotly_dark',
            width=960,
            height=540,
        )
        fig.write_image(str(output_path / f"sensitivity_{param_name}.png"),
                        width=960, height=540, scale=2)
        logger.info("敏感性图表已保存: sensitivity_%s.png", param_name)
    except Exception as e:
        logger.warning("敏感性图表生成失败: %s", e)


def run_sensitivity_suite(config: dict,
                          output_dir: str = "output/sensitivity") -> Dict[str, pd.DataFrame]:
    """
    Run full hyperparameter sensitivity suite.

    搜索空间与 tuning/optuna_search.py 的收窄原则保持一致：
    中长周期因子数据信噪比低，过宽的树复杂度容易过拟合，
    num_leaves 上限 63、max_depth 上限 6。

    Scans key LightGBM hyperparameters:
      - learning_rate: [0.01, 0.03, 0.05, 0.07, 0.10]
      - num_leaves: [15, 31, 47, 63]
      - max_depth: [3, 4, 5, 6]
      - subsample: [0.6, 0.7, 0.8, 0.9, 1.0]

    Returns:
        Dict of param_name -> sensitivity DataFrame
    """
    suite = {
        "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.10],
        "num_leaves": [15, 31, 47, 63],
        "max_depth": [3, 4, 5, 6],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    }

    # 统一初始化 Qlib + 复用 dataset 跨所有参数扫描
    _init_qlib_env(config)
    shared_dataset = None

    results = {}
    for param_name, values in suite.items():
        logger.info("-" * 40)
        # 第一个参数扫描时创建 dataset，后续参数扫描复用
        # 通过 hyperparameter_sensitivity 的 shared_dataset 参数传入
        # 但 hyperparameter_sensitivity 内部会更新 shared_dataset，所以需要拿到返回的 dataset
        # 改为直接调用 _single_train 循环以保持 dataset 复用
        param_results = []
        for val in values:
            override = {param_name: val}
            score, shared_dataset = _single_train(config, override, dataset=shared_dataset)
            if score is not None:
                param_results.append({"param_value": val, "valid_l2": score})
                logger.info("  %s=%.4f -> valid_l2=%.6f", param_name, val, score)

        df = pd.DataFrame(param_results)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if not df.empty:
            df.to_csv(output_path / f"sensitivity_{param_name}.csv", index=False)
            _generate_sensitivity_chart(param_name, df, output_path)
        results[param_name] = df

    logger.info("敏感性分析套件完成: %d 参数", len(results))
    return results