# -*- coding: utf-8 -*-
"""
Purged K-Fold Time Series Cross Validation — tscv.py

Phase 3 鲁棒性深化。
实现带 Purging 和 Embargo 的时序交叉验证（真双向净化），
基于 Marc Lopez de Prado《Advances in Financial Machine Learning》。

与简单前向验证的区别：
  前向验证: train = 所有 test 之前的数据（单向）
  真 Purged K-Fold: train 可以包含 test 之后的数据，只要标签窗口不重叠
  这能显着提升对稀缺样本的利用效率，特别适合中长周期策略。

Usage:
    from qlib_pipeline.tscv import PurgedKFoldCV, run_tscv
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Generator

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from qlib_pipeline.numpy_compat import *  # noqa

logger = logging.getLogger(__name__)


def generate_purged_kfold_splits(
    dates: List[str],
    n_splits: int = 5,
    purge_days: int = 5,
    embargo_days: int = 0,
    label_horizon: int = 20,
) -> Generator[Tuple[List[str], List[str]], None, None]:
    """
    Generate Purged K-Fold splits for time series（真双向净化）。

    Key concepts:
      - Purging: 训练集中，标签计算窗口与测试集有重叠的样本全部排除
      - Embargo: 测试集结束后留一段禁运期，不用于紧邻的下一折训练
      - 双向: 允许使用测试集之后的数据训练（只要满足净化条件），
        这是 K-Fold 区别于纯前向验证的关键

    Args:
        dates: sorted list of date strings
        n_splits: number of folds
        purge_days: 标签窗口重叠的净化天数（建议设为 label_horizon）
        embargo_days: 测试集后的禁运天数
        label_horizon: 标签的前瞻天数，用于计算标签窗口重叠

    Yields:
        (train_dates, test_dates) for each fold
    """
    n = len(dates)
    date_dt = pd.to_datetime(dates)
    purge_td = pd.Timedelta(days=purge_days)
    embargo_td = pd.Timedelta(days=embargo_days)
    label_td = pd.Timedelta(days=label_horizon)

    for i in range(n_splits):
        # 测试集: i-th block
        test_start_idx = int(n * i / n_splits)
        test_end_idx = int(n * (i + 1) / n_splits) - 1
        test_start_date = date_dt[test_start_idx]
        test_end_date = date_dt[test_end_idx]

        # 测试窗口（含 purge 和 embargo 扩展）
        test_window_start = test_start_date - purge_td
        test_window_end = test_end_date + embargo_td

        # 训练集: 所有标签窗口不与测试窗口重叠的样本
        # 训练样本在日期 d 的标签覆盖 [d, d + label_horizon]
        # 重叠条件: d + label_horizon >= test_window_start AND d <= test_window_end
        train_dates = []
        for j, d in enumerate(date_dt):
            if test_start_idx <= j <= test_end_idx:
                continue  # 跳过测试集本身
            label_end = d + label_td
            # 标签窗口与测试窗口重叠？
            if label_end < test_window_start or d > test_window_end:
                train_dates.append(dates[j])

        test_dates = [dates[i] for i in range(test_start_idx, test_end_idx + 1)]

        if len(train_dates) > 0 and len(test_dates) > 0:
            yield train_dates, test_dates
        else:
            logger.warning(
                "Fold %d: 数据不足 (train=%d, test=%d), 跳过",
                i, len(train_dates), len(test_dates),
            )


class PurgedKFoldCV:
    """
    Purged K-Fold Time Series Cross Validation（真双向净化）。

    Implements time-aware cross-validation with purging and embargo
    based on Marc Lopez de Prado's methodology.

    Args:
        config: workflow config dict
        n_splits: number of folds
        purge_days: 标签窗口重叠净化天数（建议 = label_horizon）
        embargo_days: 测试集后禁运天数
        label_horizon: 标签前瞻天数，用于计算标签窗口重叠
        min_train_days: minimum training days required
    """

    def __init__(self, config: dict, n_splits: int = 5,
                 purge_days: int = 5, embargo_days: int = 0,
                 label_horizon: int = 20,
                 min_train_days: int = 252):
        self.config = config
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.label_horizon = label_horizon
        self.min_train_days = min_train_days
        self.fold_results: List[Dict] = []

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

    def _load_calendar(self) -> List[str]:
        """Load trading calendar from Qlib data."""
        from qlib.data import D
        try:
            cal = D.calendar(start_time="2020-01-01", end_time="2025-12-31")
            return [str(c) for c in cal]
        except Exception:
            return pd.date_range("2020-01-01", "2025-12-31", freq="B").strftime("%Y-%m-%d").tolist()

    def run(self) -> pd.DataFrame:
        """
        Execute Purged K-Fold TSCV.

        Returns:
            DataFrame with per-fold metrics
        """
        from qlib.utils import init_instance_by_config
        from qlib.workflow import R
        from qlib.utils import flatten_dict

        calendar = self._load_calendar()
        splits = list(generate_purged_kfold_splits(
            calendar, self.n_splits, self.purge_days, self.embargo_days,
            label_horizon=self.label_horizon
        ))

        self._init_qlib()

        handler = self.config.get("dataset", {}).get("handler", "Alpha158")
        logger.info("Purged K-Fold TSCV: %d folds, purge=%dd, embargo=%dd",
                     len(splits), self.purge_days, self.embargo_days)

        for i, (train_dates, test_dates) in enumerate(splits):
            if len(train_dates) < self.min_train_days:
                logger.warning("Fold %d: 跳过(训练天数=%d < %d)", i, len(train_dates), self.min_train_days)
                continue

            logger.info("Fold %d: train=%s~%s (%d days), test=%s~%s (%d days)",
                         i, train_dates[0], train_dates[-1], len(train_dates),
                         test_dates[0], test_dates[-1], len(test_dates))

            fold_config = self.config.copy()
            fold_config["data_handler"] = {
                "start_time": train_dates[0],
                "end_time": test_dates[-1],
                "fit_start_time": train_dates[0],
                "fit_end_time": train_dates[-1],
                "instruments": self.config.get("data_handler", {}).get("instruments", "csi300"),
            }
            fold_config["dataset"] = {
                "handler": handler,
                "segments": {
                    "train": [train_dates[0], train_dates[-1]],
                    "test": [test_dates[0], test_dates[-1]],
                },
            }

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

                with R.start(experiment_name=f"tscv_fold_{i}"):
                    R.log_params(**flatten_dict(task))
                    R.log_params(handler_type=handler)
                    model.fit(dataset)
                    rid = R.get_recorder().id

                # 评估：在测试集上计算 IC 指标
                from qlib_pipeline.ic_stability import evaluate_fold, get_feature_importance
                eval_metrics = evaluate_fold(model, dataset)
                feat_importance = get_feature_importance(model, dataset)

                self.fold_results.append({
                    "fold": i,
                    "train_start": train_dates[0],
                    "train_end": train_dates[-1],
                    "test_start": test_dates[0],
                    "test_end": test_dates[-1],
                    "n_train_days": len(train_dates),
                    "n_test_days": len(test_dates),
                    "recorder_id": rid,
                    "status": "success",
                    "feature_importance": feat_importance,
                    **eval_metrics,
                })
                logger.info("Fold %d 完成: rid=%s, IC_mean=%.4f, ICIR=%.4f, feats=%d",
                             i, rid,
                             eval_metrics.get("ic_mean", 0) or 0,
                             eval_metrics.get("icir", 0) or 0,
                             len(feat_importance))

            except Exception as e:
                self.fold_results.append({
                    "fold": i,
                    "train_start": train_dates[0],
                    "train_end": train_dates[-1],
                    "status": "failed",
                    "error": str(e),
                })
                logger.error("Fold %d 失败: %s", i, e)

        df = pd.DataFrame(self.fold_results)
        success = len(df[df["status"] == "success"])
        logger.info("Purged K-Fold TSCV 完成: %d/%d folds 成功", success, len(df))
        return df


def run_tscv(config: dict, n_splits: int = 5, purge_days: int = 5,
             embargo_days: int = 0, label_horizon: int = 20,
             output_dir: str = "output/tscv") -> pd.DataFrame:
    """Convenience wrapper for TSCV."""
    cv = PurgedKFoldCV(config, n_splits=n_splits, purge_days=purge_days,
                       embargo_days=embargo_days, label_horizon=label_horizon)
    df = cv.run()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path / "tscv_results.csv", index=False)
    logger.info("TSCV 结果已保存到: %s", output_path / "tscv_results.csv")

    return df