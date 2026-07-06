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
from typing import Dict, List, Tuple, Optional

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

    import qlib
    from qlib.constant import REG_CN
    provider_uri = config.get("qlib", {}).get("provider_uri",
                     "D:/trae/qlib_bin")
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    from qlib.config import C
    C.joblib_backend = "threading"
    C.dataset_process_n_worker = 1


def _single_train(config: dict, param_overrides: dict) -> Optional[float]:
    """Train a single model with given params, return final valid l2."""
    import os
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

    from qlib.utils import init_instance_by_config
    from qlib.workflow import R
    from qlib.utils import flatten_dict

    import qlib
    from qlib.constant import REG_CN
    provider_uri = config.get("qlib", {}).get("provider_uri",
                     "D:/trae/qlib_bin")
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    from qlib.config import C
    C.joblib_backend = "threading"
    C.dataset_process_n_worker = 1

    handler = config.get("dataset", {}).get("handler", "Alpha158")

    # Merge param overrides into model config
    model_kwargs = config.get("qlib_lgb", {}).get("kwargs", {}).copy()
    model_kwargs.update(param_overrides)

    handler_cfg = {
        "class": handler,
        "module_path": "qlib.contrib.data.handler",
        "kwargs": config.get("data_handler", {}),
    }

    task = {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": model_kwargs,
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": handler_cfg,
                "segments": config.get("dataset", {}).get("segments", {}),
            },
        },
    }

    try:
        dataset = init_instance_by_config(task["dataset"])
        model = init_instance_by_config(task["model"])

        with R.start(experiment_name="sensitivity_scan"):
            model.fit(dataset)
            # Extract best valid l2 from model's evals_result (Qlib LGBModel uses evals_result, not sklearn-style evals_result_)
            if hasattr(model, "evals_result") and "valid" in model.evals_result:
                valid_scores = model.evals_result.get("valid", {}).get("l2", [])
                if valid_scores:
                    return float(min(valid_scores))
            elif hasattr(model, "evals_result_") and "valid" in model.evals_result_:
                valid_scores = model.evals_result_.get("valid", {}).get("l2", [])
                if valid_scores:
                    return float(min(valid_scores))
            logger.warning("无法从模型中提取 valid l2 分数，敏感性得分将退化为 0.0，请检查 Qlib 版本兼容性")
            return 0.0
    except Exception as e:
        logger.warning("训练失败 (%s): %s", param_overrides, e)
        return None


def hyperparameter_sensitivity(config: dict, param_name: str,
                               values: List[float],
                               output_dir: str = "output/sensitivity") -> pd.DataFrame:
    """
    Single-variable hyperparameter sensitivity scan.

    Scans a range of values for one hyperparameter, recording the
    final valid l2 score for each.

    Args:
        config: workflow config dict
        param_name: parameter name to scan (e.g., "learning_rate", "num_leaves")
        values: list of values to try
        output_dir: output directory

    Returns:
        DataFrame with columns: param_value, valid_l2
    """
    _init_qlib_env(config)

    logger.info("超参数敏感性扫描: %s, values=%s", param_name, values)

    results = []
    for val in values:
        override = {param_name: val}
        score = _single_train(config, override)
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

    Scans key LightGBM hyperparameters:
      - learning_rate: [0.01, 0.03, 0.05, 0.07, 0.10]
      - num_leaves: [32, 64, 128, 256, 512]
      - max_depth: [4, 6, 8, 10, 12]
      - subsample: [0.6, 0.7, 0.8, 0.9, 1.0]

    Returns:
        Dict of param_name -> sensitivity DataFrame
    """
    suite = {
        "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.10],
        "num_leaves": [32, 64, 128, 256, 512],
        "max_depth": [4, 6, 8, 10, 12],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    }

    results = {}
    for param_name, values in suite.items():
        logger.info("-" * 40)
        df = hyperparameter_sensitivity(config, param_name, values, output_dir)
        results[param_name] = df

    logger.info("敏感性分析套件完成: %d 参数", len(results))
    return results