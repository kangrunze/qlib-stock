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


# LightGBM 超参白名单：用于从已构造的 Qlib LGBModel 实例尽力提取构造参数。
# 不同 Qlib 版本下 LGBModel 可能将 init 参数存为实例属性、或打包进 self.params，
# 此处覆盖 optuna_search._objective 的搜索空间 + workflow_config 的基础参数。
_LGB_PARAM_KEYS = frozenset({
    "loss", "learning_rate", "num_leaves", "max_depth", "subsample",
    "colsample_bytree", "lambda_l1", "lambda_l2", "min_child_samples",
    "num_threads", "seed", "n_estimators", "early_stopping_rounds",
})


def _extract_model_kwargs(model, config: dict) -> Tuple[dict, bool]:
    """尽力从已构造的 model 实例提取 LGB 超参（用于传播 Optuna trial 超参）。

    Args:
        model: 已构造（可能已训练）的 Qlib LGBModel 实例
        config: workflow config，作为回退基线

    Returns:
        (merged_kwargs, extracted): merged_kwargs 为合并后的 kwargs；
        extracted=True 表示从 model 实例提取到了至少一个白名单超参
        （意味着 trial 超参已传播）；False 表示只能回退到 config 基线，
        此时调用方应放弃滚动 CV 以避免 trial 超参丢失。
    """
    base = dict(config.get("qlib_lgb", {}).get("kwargs", {}))
    if model is None:
        return base, False

    # 路径1：实例属性（Qlib LGBModel 常见模式：__init__ 将超参存为 self.<name>）
    try:
        attrs = vars(model)
    except TypeError:
        attrs = None
    if attrs:
        overrides = {k: v for k, v in attrs.items() if k in _LGB_PARAM_KEYS}
        if overrides:
            base.update(overrides)
            return base, True

    # 路径2：model.params dict（lgb Booster 风格）
    params = getattr(model, "params", None)
    if isinstance(params, dict) and params:
        overrides = {k: v for k, v in params.items() if k in _LGB_PARAM_KEYS}
        if overrides:
            base.update(overrides)
            return base, True

    # 路径3：回退到 config 基线（trial 超参无法传播）
    return base, False


def _label_horizon_from_config(config: dict) -> int:
    """从 config['labels']['primary'] 解析标签周期（天），用于 NW max_lags。"""
    import re
    primary = config.get("labels", {}).get("primary", "ret_20d")
    m = re.match(r"^(?:ret|xs_ret|alpha|up_down)_(\d+)d$", primary)
    return int(m.group(1)) if m else 20


def rolling_cross_validation(
    config: dict,
    model=None,
    dataset=None,
    n_folds: int = 3,
    window_years: float = 3.0,
    step_months: int = 6,
    test_years: float = 1.0,
) -> Dict:
    """轻量级滚动交叉验证，专供 Optuna 调参场景使用。

    与 RollingTrainer.run() 的区别：
      - 不走 Qlib Recorder / MLflow 记录流程（调参需要速度，无需每折存一条记录）
      - 仅返回 IC 汇总，不保存特征重要性等附加产物
      - 复用 generate_rolling_windows 切分窗口、evaluate_fold 计算 IC

    trial 超参传播说明：
      Optuna 调用方传入的 `model` 已用 trial 超参构造（可能已 fit），
      但 `config` 是原始 config（未合并 trial 超参）。本函数尽力从 `model`
      实例提取构造超参以传播到每折；若提取失败（Qlib LGBModel 未暴露
      构造参数），将抛出 RuntimeError，由调用方 except 捕获后回退到
      单切分评估——这避免了"用基线超参跑滚动 CV 导致所有 trial IC 相同"
      的回归风险。

    Args:
        config: 完整 workflow 配置字典（包含 qlib/data_handler/dataset 段）
        model: 已构造的 Qlib LGBModel 实例（用于提取 trial 超参；
               其训练状态在滚动 CV 中不可复用，每折从 config 重建）
        dataset: 兼容签名而保留（每折从 config 重新构造 DatasetH）
        n_folds: 滚动窗口数
        window_years: 训练窗口年数
        step_months: 窗口步长月数
        test_years: 测试窗口年数

    Returns:
        dict: {
            "fold_ics": [
                {"fold": int, "ic_mean": float|None, "ic_std": float,
                 "icir": float, "n_samples": int,
                 "status": "success"|"failed", "error"?: str},
                ...
            ],
            "n_success": int,
            "n_folds": int,
            "mean_ic": float|None,   # 成功折的 IC 均值
        }
        失败折也会被记录（status="failed", ic_mean=None），但不参与 mean_ic。
        若无法从 model 提取 trial 超参，抛 RuntimeError 促使调用方回退单切分。
    """
    # 尽力从传入 model 提取 trial 超参；提取不到则抛错让调用方回退单切分。
    # 此检查是纯 Python 逻辑，放在 qlib import 之前——提取失败时无需加载 qlib
    # 即可快速失败，由 optuna 的 except 捕获后回退到单切分评估。
    model_kwargs, extracted = _extract_model_kwargs(model, config)
    if not extracted:
        raise RuntimeError(
            "无法从传入的 model 实例提取 LGB 构造超参（Qlib LGBModel 未将 init 参数"
            "暴露为实例属性），trial 超参无法传播到滚动 CV；回退到单切分评估以避免"
            "所有 trial 使用相同基线超参导致 Optuna 失效"
        )

    from qlib.utils import init_instance_by_config
    from qlib_pipeline.ic_stability import evaluate_fold

    dh = config.get("data_handler", {})
    start = dh.get("start_time", "2020-01-01")
    end = dh.get("end_time", "2025-12-31")

    windows = generate_rolling_windows(
        start, end, train_years=window_years,
        step_months=step_months, test_years=test_years,
    )
    if n_folds:
        windows = windows[:n_folds]

    if len(windows) == 0:
        raise RuntimeError(
            f"generate_rolling_windows 在 [{start}, {end}] 下未产出任何窗口"
            f"(window_years={window_years}, step_months={step_months})"
        )

    handler = config.get("dataset", {}).get("handler", "Alpha158")
    instruments = dh.get("instruments", "csi300")
    model_cfg = config.get("qlib_lgb", {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {"loss": "mse", "num_threads": 20},
    })
    label_horizon = _label_horizon_from_config(config)

    logger.debug(
        "Rolling CV: %d 折 (window=%.1fyr, step=%dmo, test=%.1fyr), trial 超参已传播",
        len(windows), window_years, step_months, test_years,
    )

    fold_ics = []
    n_success = 0
    sum_ic = 0.0

    for w in windows:
        fold_handler_kwargs = {
            "start_time": w["train"][0],
            "end_time": w["test"][1],
            "fit_start_time": w["train"][0],
            "fit_end_time": w["train"][1],
            "instruments": instruments,
        }
        fold_task = {
            "model": {
                "class": model_cfg.get("class", "LGBModel"),
                "module_path": model_cfg.get("module_path", "qlib.contrib.model.gbdt"),
                "kwargs": model_kwargs,
            },
            "dataset": {
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": {
                        "class": handler,
                        "module_path": "qlib.contrib.data.handler",
                        "kwargs": fold_handler_kwargs,
                    },
                    "segments": {
                        "train": w["train"],
                        "valid": w["valid"],
                        "test": w["test"],
                    },
                },
            },
        }

        fold_entry: Dict = {
            "fold": w["fold"],
            "train_start": w["train"][0],
            "train_end": w["train"][1],
            "test_start": w["test"][0],
            "test_end": w["test"][1],
            "ic_mean": None,
            "status": "failed",
        }

        try:
            fold_dataset = init_instance_by_config(fold_task["dataset"])
            fold_model = init_instance_by_config(fold_task["model"])
            fold_model.fit(fold_dataset)

            metrics = evaluate_fold(
                fold_model, fold_dataset, label_horizon=label_horizon
            )
            ic_mean = metrics.get("ic_mean")
            fold_entry.update({
                "ic_mean": ic_mean,
                "ic_std": metrics.get("ic_std"),
                "icir": metrics.get("icir"),
                "n_samples": metrics.get("n_samples", 0),
                "status": "success" if ic_mean is not None else "failed",
            })
            if ic_mean is not None:
                n_success += 1
                sum_ic += float(ic_mean)
        except Exception as e:
            logger.warning("Rolling CV fold %d 失败: %s", w["fold"], e)
            fold_entry["error"] = str(e)

        fold_ics.append(fold_entry)

    mean_ic = (sum_ic / n_success) if n_success > 0 else None

    logger.debug(
        "Rolling CV 完成: %d/%d 折成功, mean_ic=%s",
        n_success, len(windows), mean_ic,
    )

    return {
        "fold_ics": fold_ics,
        "n_success": n_success,
        "n_folds": len(windows),
        "mean_ic": mean_ic,
    }