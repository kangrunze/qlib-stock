# -*- coding: utf-8 -*-
"""
滚动训练与 Walk Forward 验证 — rolling.py

Phase 1 稳健性基础设施核心模块。
实现 RollingDataHandler 滚动训练 + Walk Forward 验证，
评估模型在时间维度上的泛化稳定性。

Usage:
    from qlib_pipeline.rolling import RollingTrainer, walk_forward_validate
    trainer = RollingTrainer(config, n_folds=6, window=3)
    results = trainer.run()
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from qlib_pipeline.numpy_compat import *  # noqa

logger = logging.getLogger(__name__)


def _month_offset(date_str: str, months: int) -> str:
    """Add months to a date string YYYY-MM-DD."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    m = dt.month + months
    y = dt.year + (m - 1) // 12
    m = (m - 1) % 12 + 1
    d = min(dt.day, 28)
    return f"{y:04d}-{m:02d}-{d:02d}"


def _quarter_offset(date_str: str, quarters: int) -> str:
    return _month_offset(date_str, quarters * 3)


def generate_rolling_windows(
    start: str, end: str, train_years: float = 3.0,
    step_months: int = 6, test_years: float = 1.0
) -> List[Dict[str, str]]:
    """
    Generate rolling walk-forward windows.

    Args:
        start: overall start date YYYY-MM-DD
        end: overall end date YYYY-MM-DD
        train_years: training window in years
        step_months: step size in months
        test_years: test window in years

    Returns:
        List of dicts with train/valid/test date ranges
    """
    windows = []
    train_start = start
    train_months = int(train_years * 12)
    test_months = int(test_years * 12)

    while True:
        train_end = _month_offset(train_start, train_months)
        valid_start = train_end
        valid_end = _month_offset(valid_start, test_months)
        test_start = valid_end
        test_end = _month_offset(test_start, test_months)

        if test_start >= end:
            break

        # Ensure valid_end doesn't exceed end
        if valid_end > end:
            valid_end = end
        if test_end > end:
            test_end = end

        windows.append({
            "train": [train_start, train_end],
            "valid": [valid_start, valid_end],
            "test": [test_start, test_end],
            "fold": len(windows),
        })

        train_start = _month_offset(train_start, step_months)

    return windows


class RollingTrainer:
    """
    Rolling walk-forward trainer.

    Iterates through rolling windows, trains a model on each fold,
    and collects performance metrics across folds.

    Args:
        config: workflow config dict
        n_folds: number of rolling folds (auto if None)
        window_years: training window size in years
        step_months: step size between folds
        test_years: test window size in years
    """

    def __init__(self, config: dict, n_folds: Optional[int] = None,
                 window_years: float = 3.0, step_months: int = 6,
                 test_years: float = 1.0):
        self.config = config
        self.n_folds = n_folds
        self.window_years = window_years
        self.step_months = step_months
        self.test_years = test_years
        self.results: List[Dict] = []

    def _get_overall_dates(self) -> Tuple[str, str]:
        dh = self.config.get("data_handler", {})
        return dh.get("start_time", "2020-01-01"), dh.get("end_time", "2025-12-31")

    def _init_qlib(self):
        import os
        os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        import qlib
        from qlib.constant import REG_CN
        provider_uri = self.config.get("qlib", {}).get("provider_uri",
                         "D:/trae/qlib_bin")
        qlib.init(provider_uri=provider_uri, region=REG_CN)

        from qlib.config import C
        C.joblib_backend = "threading"
        C.dataset_process_n_worker = 1

    def run(self) -> pd.DataFrame:
        """
        Execute rolling walk-forward training.

        Returns:
            DataFrame with columns: fold, train_period, valid_period, test_period,
            train_l2, valid_l2, best_iteration
        """
        from qlib.utils import init_instance_by_config
        from qlib.workflow import R
        from qlib.utils import flatten_dict

        start, end = self._get_overall_dates()
        windows = generate_rolling_windows(
            start, end, self.window_years, self.step_months, self.test_years
        )

        if self.n_folds:
            windows = windows[:self.n_folds]

        self._init_qlib()

        logger.info("Rolling Walk-Forward: %d folds, %.1fyr window, %dmo step",
                     len(windows), self.window_years, self.step_months)

        handler = self.config.get("dataset", {}).get("handler", "Alpha158")

        for w in windows:
            logger.info("-" * 50)
            logger.info("Fold %d: train=%s~%s, valid=%s~%s, test=%s~%s",
                         w["fold"], w["train"][0], w["train"][1],
                         w["valid"][0], w["valid"][1],
                         w["test"][0], w["test"][1])

            # Build fold-specific config
            fold_config = self.config.copy()
            fold_config["data_handler"] = {
                "start_time": w["train"][0],
                "end_time": w["test"][1],
                "fit_start_time": w["train"][0],
                "fit_end_time": w["train"][1],
                "instruments": self.config.get("data_handler", {}).get("instruments", "csi300"),
            }
            fold_config["dataset"] = {
                "handler": handler,
                "segments": {
                    "train": w["train"],
                    "valid": w["valid"],
                    "test": w["test"],
                }
            }

            # Build task
            handler_cfg = {
                "class": handler,
                "module_path": "qlib.contrib.data.handler",
                "kwargs": fold_config["data_handler"],
            }

            model_cfg = self.config.get("qlib_lgb", {
                "class": "LGBModel",
                "module_path": "qlib.contrib.model.gbdt",
                "kwargs": {"loss": "mse", "num_threads": 20},
            })

            task = {
                "model": {
                    "class": model_cfg.get("class", "LGBModel"),
                    "module_path": model_cfg.get("module_path", "qlib.contrib.model.gbdt"),
                    "kwargs": model_cfg.get("kwargs", {}),
                },
                "dataset": {
                    "class": "DatasetH",
                    "module_path": "qlib.data.dataset",
                    "kwargs": {
                        "handler": handler_cfg,
                        "segments": fold_config["dataset"]["segments"],
                    },
                },
            }

            try:
                dataset = init_instance_by_config(task["dataset"])
                model = init_instance_by_config(task["model"])

                with R.start(experiment_name=f"rolling_fold_{w['fold']}"):
                    R.log_params(**flatten_dict(task))
                    R.log_params(handler_type=handler)
                    model.fit(dataset)
                    rid = R.get_recorder().id

                # 评估：在测试集上计算 IC 指标
                from qlib_pipeline.ic_stability import evaluate_fold, get_feature_importance
                eval_metrics = evaluate_fold(model, dataset)
                feat_importance = get_feature_importance(model, dataset)

                fold_result = {
                    "fold": w["fold"],
                    "train_start": w["train"][0],
                    "train_end": w["train"][1],
                    "valid_start": w["valid"][0],
                    "valid_end": w["valid"][1],
                    "test_start": w["test"][0],
                    "test_end": w["test"][1],
                    "recorder_id": rid,
                    "status": "success",
                    "feature_importance": feat_importance,
                    **eval_metrics,
                }
                logger.info("Fold %d 完成: rid=%s, IC_mean=%.4f, ICIR=%.4f, feats=%d",
                             w["fold"], rid,
                             eval_metrics.get("ic_mean", 0) or 0,
                             eval_metrics.get("icir", 0) or 0,
                             len(feat_importance))

            except Exception as e:
                fold_result = {
                    "fold": w["fold"],
                    "train_start": w["train"][0],
                    "train_end": w["train"][1],
                    "status": "failed",
                    "error": str(e),
                }
                logger.error("Fold %d 失败: %s", w["fold"], e)

            self.results.append(fold_result)

        df = pd.DataFrame(self.results)
        logger.info("Rolling Walk-Forward 完成: %d/%d folds 成功",
                     len(df[df["status"] == "success"]), len(df))
        return df


def walk_forward_validate(config: dict, n_folds: int = 6,
                          window_years: float = 3.0, step_months: int = 6,
                          output_dir: str = "output/rolling") -> pd.DataFrame:
    """
    Convenience wrapper for walk-forward validation.

    Returns the results DataFrame and generates a summary report.
    """
    trainer = RollingTrainer(
        config, n_folds=n_folds, window_years=window_years,
        step_months=step_months
    )
    df = trainer.run()

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path / "rolling_results.csv", index=False)
    logger.info("滚动训练结果已保存到: %s", output_path / "rolling_results.csv")

    return df