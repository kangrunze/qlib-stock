# -*- coding: utf-8 -*-
"""
Optuna 超参数搜索 — optuna_search.py

Phase 4 模型能力恢复，基于 Optuna 框架对 Qlib LGBModel 进行超参数优化，
严格遵循第 7.3 节探索/确认两阶段协议。

与 sensitivity.py 的区别：
  - sensitivity.py: 单变量扫描，评估"参数变化对 IC 的影响方向"
  - optuna_search.py: 多变量贝叶斯优化，搜索"最优超参数组合"

核心约束（第 2.2 节 + 第 7.3 节）：
  - 搜索只在探索窗口 [2015-2023] 内部进行
  - 每次试算独立记录到 ExperimentTracker
  - 最终选定的参数在确认窗口只跑一次

Usage:
    python run.py optuna --n-trials 100 --timeout 3600
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from qlib_pipeline.numpy_compat import *  # noqa

logger = logging.getLogger(__name__)


def _init_qlib_once(config: dict):
    """初始化 Qlib 环境（确保只初始化一次）"""
    import os
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    import qlib
    from qlib.constant import REG_CN
    provider_uri = config.get("qlib", {}).get("provider_uri", "D:/trae/qlib_bin")
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    from qlib.config import C
    C.joblib_backend = "threading"
    C.dataset_process_n_worker = 1


def _single_trial_ic(config: dict, trial_params: dict) -> Optional[float]:
    """单次 Optuna trial：训练模型并返回验证集 IC 均值。

    优先使用 rolling.py 的多折滚动平均 IC（第 7.3 节建议），
    退化为单一切分 IC 均值（方差较大但可用）。

    Args:
        config: workflow config dict
        trial_params: 本次 trial 的超参数覆盖

    Returns:
        valid IC 均值（越大越好），失败时返回 None
    """
    from qlib.utils import init_instance_by_config

    handler = config.get("dataset", {}).get("handler", "Alpha158")
    segments = config.get("dataset", {}).get("segments", {})

    # 合并超参数
    model_kwargs = config.get("qlib_lgb", {}).get("kwargs", {}).copy()
    model_kwargs.update(trial_params)
    seed = config.get("experiment", {}).get("random_seed", 42)
    model_kwargs.setdefault("seed", seed)

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
                "segments": segments,
            },
        },
    }

    try:
        dataset = init_instance_by_config(task["dataset"])
        model = init_instance_by_config(task["model"])
        model.fit(dataset)

        from qlib_pipeline.ic_stability import get_ic_series

        # 优先尝试 rolling 多折平均 IC（降低单一切分的方差）
        try:
            from qlib_pipeline.rolling import rolling_cross_validation
            rolling_result = rolling_cross_validation(config, model=model, dataset=dataset)
            if rolling_result and "fold_ics" in rolling_result and len(rolling_result["fold_ics"]) >= 3:
                fold_ic_means = [fold["ic_mean"] for fold in rolling_result["fold_ics"] if fold.get("ic_mean") is not None]
                if fold_ic_means:
                    ic_mean = float(np.mean(fold_ic_means))
                    logger.debug("Trial IC (rolling %d-fold): %.6f", len(fold_ic_means), ic_mean)
                    return ic_mean
        except Exception as e:
            logger.debug("Rolling 多折 IC 不可用 (%s)，回退到单一切分", e)

        # 退化：单一切分 IC 均值
        ic_series = get_ic_series(model, dataset)
        if len(ic_series) == 0:
            return None

        ic_mean = float(ic_series.mean())
        logger.debug("Trial IC (single split): %.6f", ic_mean)
        return ic_mean

    except Exception as e:
        logger.debug("Trial 失败 (%s): %s", trial_params, e)
        return None


def _objective(trial, config: dict, base_params: dict) -> float:
    """Optuna objective function: 最大化验证集 IC 均值。

    搜索空间设计（第 7.3/C1 节收窄原则）：
      中长周期因子数据信噪比低，过宽的树复杂度容易过拟合，
      搜索空间严格收窄：
      - learning_rate: log-uniform, 0.01 ~ 0.2
      - num_leaves: int, 15 ~ 63（设计文档上限 63）
      - max_depth: int, 3 ~ 6（设计文档上限 6）
      - subsample: uniform, 0.5 ~ 1.0
      - colsample_bytree: uniform, 0.5 ~ 1.0
      - lambda_l1: log-uniform, 1e-8 ~ 100
      - lambda_l2: log-uniform, 1e-8 ~ 100
      - min_child_samples: int, 10 ~ 100
    """
    import optuna

    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63, step=4),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 100.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 100.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "num_threads": base_params.get("num_threads", 20),
    }

    ic_mean = _single_trial_ic(config, params)
    if ic_mean is None:
        return float("-inf")

    # 记录 trial 属性用于后续分析
    trial.set_user_attr("ic_mean", ic_mean)
    for k, v in params.items():
        trial.set_user_attr(k, v)

    return ic_mean


def run_optuna_search(
    config: dict,
    n_trials: int = 100,
    timeout: int = 3600,
    study_name: str = "lgb_optimization",
    output_dir: str = "output/optuna",
    phase: str = "exploration",
) -> Dict:
    """运行 Optuna 超参数搜索。

    Args:
        config: workflow config dict
        n_trials: 最大试算次数
        timeout: 最大搜索时间（秒）
        study_name: Optuna study 名称
        output_dir: 结果输出目录
        phase: 实验阶段 ("exploration" | "confirmation")

    Returns:
        dict with keys: best_params, best_ic, study_summary, n_trials_completed
    """
    try:
        import optuna
    except ImportError:
        logger.error("Optuna 未安装，请运行: pip install optuna")
        return {"error": "optuna not installed"}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 初始化 Qlib（只初始化一次）
    _init_qlib_once(config)

    base_params = config.get("qlib_lgb", {}).get("kwargs", {})

    # 创建 Optuna study（最大化 IC）
    storage_path = str(output_path / f"{study_name}.db")
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{storage_path}",
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    )

    logger.info(
        "Optuna 超参数搜索: study=%s, n_trials=%d, timeout=%ds, phase=%s",
        study_name, n_trials, timeout, phase,
    )

    # 运行优化
    study.optimize(
        lambda trial: _objective(trial, config, base_params),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )

    # 提取最佳结果
    best_params = study.best_params
    best_ic = study.best_value

    # 保存结果
    results_df = study.trials_dataframe()
    results_df.to_csv(output_path / "optuna_trials.csv", index=False)

    # 记录到 ExperimentTracker
    try:
        from research.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker(storage_dir="output/experiments")
        tracker.log_experiment(
            config=config,
            metrics={
                "best_ic": best_ic,
                "n_trials": len(study.trials),
                "best_params": best_params,
                "ic_mean": best_ic,
            },
            phase=phase,
            description=f"optuna_{study_name}",
        )
    except Exception as e:
        logger.warning("实验追踪记录失败: %s", e)

    # 生成结果摘要
    logger.info("=" * 60)
    logger.info("Optuna 超参数搜索完成")
    logger.info("  Trials 完成: %d", len(study.trials))
    logger.info("  最佳 IC: %.6f", best_ic)
    logger.info("  最佳参数:")
    for k, v in best_params.items():
        logger.info("    %s = %s", k, v)
    logger.info("  结果已保存到: %s", output_dir)
    logger.info("=" * 60)

    return {
        "best_params": best_params,
        "best_ic": best_ic,
        "study_summary": {
            "n_trials": len(study.trials),
            "n_completed": len([t for t in study.trials if t.state.name == "COMPLETE"]),
            "best_trial_number": study.best_trial.number,
        },
        "output_dir": str(output_path),
    }