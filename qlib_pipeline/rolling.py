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

import calendar
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from qlib_pipeline.numpy_compat import *  # noqa

logger = logging.getLogger(__name__)


def _month_offset(date_str: str, months: int) -> str:
    """Add months to a date string YYYY-MM-DD.

    月末日期会正确映射到目标月的最后一天（如 1/31 + 1月 → 2/28或29），
    而非一律截断为 28 号。
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    m = dt.month + months
    y = dt.year + (m - 1) // 12
    m = (m - 1) % 12 + 1
    # 用目标年月实际最后一天，避免丢失 29-31 号
    last_day_of_target_month = calendar.monthrange(y, m)[1]
    d = min(dt.day, last_day_of_target_month)
    return f"{y:04d}-{m:02d}-{d:02d}"


def _quarter_offset(date_str: str, quarters: int) -> str:
    return _month_offset(date_str, quarters * 3)


def generate_rolling_windows(
    start: str, end: str, train_years: float = 3.0,
    step_months: int = 6, test_years: float = 1.0,
    label_horizon: int = 0,
) -> List[Dict[str, str]]:
    """
    Generate rolling walk-forward windows.

    Args:
        start: overall start date YYYY-MM-DD
        end: overall end date YYYY-MM-DD
        train_years: training window in years
        step_months: step size in months
        test_years: test window in years
        label_horizon: 标签前瞻天数（如 60 日收益率标签 = 60）。
            当 > 0 时，在 train/valid 和 valid/test 边界处留 purge gap，
            防止训练集标签窗口（如 ret_60d 覆盖未来 60 天）延伸到
            验证/测试期导致数据泄露。
            gap 大小 = label_horizon 个交易日折算为日历日（×7/5），
            确保 60 交易日标签 → 84 日历日 gap，完全覆盖标签窗口。

    Returns:
        List of dicts with train/valid/test date ranges
    """
    windows = []
    train_start = start
    train_months = int(train_years * 12)
    test_months = int(test_years * 12)
    # purge gap: label_horizon 个交易日 → 折算为日历日（×7/5）
    #   60 交易日 ≈ 84 日历日，确保标签窗口完全被 gap 覆盖
    gap_days = pd.Timedelta(days=int(label_horizon * 7 / 5)) if label_horizon > 0 else None

    while True:
        train_end_raw = _month_offset(train_start, train_months)
        valid_start_raw = _month_offset(train_end_raw, 0)  # = train_end_raw
        valid_end_raw = _month_offset(valid_start_raw, test_months)
        test_start_raw = valid_end_raw
        test_end = _month_offset(test_start_raw, test_months)

        if test_start_raw >= end:
            break

        # 应用 purge gap：
        #   train_end 往前收缩 gap，使训练集最后一天 + label_horizon < valid_start
        #   valid_end 往前收缩 gap，使验证集最后一天 + label_horizon < test_start
        if gap_days is not None:
            train_end = str(pd.Timestamp(train_end_raw) - gap_days)[:10]
            valid_start = str(pd.Timestamp(valid_start_raw) - gap_days)[:10]
            valid_end = str(pd.Timestamp(valid_end_raw) - gap_days)[:10]
            test_start = str(pd.Timestamp(test_start_raw) - gap_days)[:10]
        else:
            train_end = train_end_raw
            valid_start = valid_start_raw
            valid_end = valid_end_raw
            test_start = test_start_raw

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
        # 从 config 解析 label_horizon，用于 walk-forward purge gap
        from qlib_pipeline.tscv import _resolve_label_horizon
        self.label_horizon = _resolve_label_horizon(config)
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
        provider_uri = os.environ.get("QLIB_PROVIDER_URI") or self.config.get("qlib", {}).get("provider_uri")
        if not provider_uri:
            raise ValueError("未设置 QLIB_PROVIDER_URI 环境变量，且配置文件中也未指定 qlib.provider_uri")
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
            start, end, self.window_years, self.step_months, self.test_years,
            label_horizon=self.label_horizon,
        )

        if self.n_folds:
            windows = windows[:self.n_folds]

        self._init_qlib()

        logger.info("Rolling Walk-Forward: %d folds, %.1fyr window, %dmo step",
                     len(windows), self.window_years, self.step_months)

        handler = self.config.get("dataset", {}).get("handler", "Alpha158")

        from qlib_pipeline.train import build_task
        import copy as _copy

        for w in windows:
            logger.info("-" * 50)
            logger.info("Fold %d: train=%s~%s, valid=%s~%s, test=%s~%s",
                         w["fold"], w["train"][0], w["train"][1],
                         w["valid"][0], w["valid"][1],
                         w["test"][0], w["test"][1])

            # 深拷贝原 config，只更新时间字段，保留 label/processors/freq/qlib_lgb 等
            # 这样 build_task() 会正确处理标签表达式、CSMedianSubtract、RankLGBModel 路由
            fold_config = _copy.deepcopy(self.config)
            # 更新 data_handler 的时间范围（保留其他字段如 label/instruments/processors）
            fold_config.setdefault("data_handler", {})
            fold_config["data_handler"]["start_time"] = w["train"][0]
            fold_config["data_handler"]["end_time"] = w["test"][1]
            fold_config["data_handler"]["fit_start_time"] = w["train"][0]
            fold_config["data_handler"]["fit_end_time"] = w["train"][1]
            # 更新 dataset.segments 为该折的时间窗口
            fold_config.setdefault("dataset", {})
            fold_config["dataset"]["handler"] = handler
            fold_config["dataset"]["segments"] = {
                "train": w["train"],
                "valid": w["valid"],
                "test": w["test"],
            }

            # 通过 build_task 构建 task dict，确保与主管线逻辑完全一致
            task = build_task(fold_config)

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


def rolling_cross_validation(
    config: dict,
    model=None,
    dataset=None,
    n_folds: int = 6,
    window_years: float = 3.0,
    step_months: int = 6,
    test_years: float = 1.0,
) -> Dict:
    """轻量级多折滚动交叉验证（供 Optuna 调参场景使用）。

    与 RollingTrainer.run() 的区别：
      - 不走 MLflow Recorder 记录流程（速度快，适合调参场景）
      - 直接返回 IC 指标字典，与 optuna_search.py 的调用约定匹配
      - 复用传入 dataset 的 handler 实例（避免每折重新 fetch 特征数据）

    每折内部逻辑：
      1. 复用 generate_rolling_windows() 切分时间窗口
      2. 用 init_instance_by_config 构建新 DatasetH（segments 为该折的 train/valid/test）
      3. 用 init_instance_by_config 构建新 model 实例并 fit
      4. 调用 evaluate_fold() 计算 IC 指标（含 Newey-West 显著性）

    Args:
        config: 完整 workflow 配置字典
        model: 调用者已构造的模型实例（保留参数以匹配调用约定，
               本函数内部每折会重新构建并训练新模型实例，不直接使用此参数）
        dataset: 已构造的 DatasetH 实例（用于复用其 handler，加速特征准备；
                 为 None 时每折完整重建 handler）
        n_folds: 最大折数
        window_years: 训练窗口年数
        step_months: 步长（月）
        test_years: 测试窗口年数

    Returns:
        dict with key "fold_ics"，值为 list[dict]，每个 dict 至少包含 "ic_mean" 键。
        滚动窗口不足或全部失败时返回 {"fold_ics": []}。
    """
    from qlib.utils import init_instance_by_config
    from qlib_pipeline.ic_stability import evaluate_fold

    # 从 config 提取整体时间范围（与 RollingTrainer._get_overall_dates 保持一致）
    dh_cfg = config.get("data_handler", {})
    start = dh_cfg.get("start_time", "2020-01-01")
    end = dh_cfg.get("end_time", "2025-12-31")

    # 从 config 解析 label_horizon，用于 walk-forward purge gap
    from qlib_pipeline.tscv import _resolve_label_horizon
    label_horizon = _resolve_label_horizon(config)

    windows = generate_rolling_windows(
        start, end, window_years, step_months, test_years,
        label_horizon=label_horizon,
    )
    if n_folds:
        windows = windows[:n_folds]

    if not windows:
        logger.warning("rolling_cross_validation: 无可用滚动窗口 (start=%s, end=%s)", start, end)
        return {"fold_ics": []}

    logger.debug("rolling_cross_validation: %d folds, %.1fyr window, %dmo step",
                 len(windows), window_years, step_months)

    # 复用传入 dataset 的 handler 实例（避免每折重新 fetch 特征数据）
    reuse_handler = None
    if dataset is not None:
        try:
            reuse_handler = dataset.handler
        except Exception:
            reuse_handler = None

    handler_name = config.get("dataset", {}).get("handler", "Alpha158")
    instruments = dh_cfg.get("instruments", "csi300")

    fold_ics: List[Dict] = []
    import copy as _copy
    from qlib_pipeline.train import build_task

    for w in windows:
        fold_segments = {
            "train": w["train"],
            "valid": w["valid"],
            "test": w["test"],
        }

        try:
            # 构建 fold dataset
            if reuse_handler is not None:
                # 复用已 fit 的 handler 实例（handler 覆盖整个时间段，只需切分 segments）
                fold_dataset = init_instance_by_config({
                    "class": "DatasetH",
                    "module_path": "qlib.data.dataset",
                    "kwargs": {
                        "handler": reuse_handler,
                        "segments": fold_segments,
                    },
                })
            else:
                # fallback: 通过 build_task 构建，确保 label/processors/RankLGBModel 路由正确
                fold_config = _copy.deepcopy(config)
                fold_config.setdefault("data_handler", {})
                fold_config["data_handler"]["start_time"] = w["train"][0]
                fold_config["data_handler"]["end_time"] = w["test"][1]
                fold_config["data_handler"]["fit_start_time"] = w["train"][0]
                fold_config["data_handler"]["fit_end_time"] = w["train"][1]
                fold_config.setdefault("dataset", {})
                fold_config["dataset"]["handler"] = handler_name
                fold_config["dataset"]["segments"] = fold_segments
                fold_task = build_task(fold_config)
                fold_dataset = init_instance_by_config(fold_task["dataset"])

            # 构建 model 实例（通过 build_task 确保 loss=rank 路由正确）
            fold_config_for_model = _copy.deepcopy(config)
            fold_config_for_model.setdefault("dataset", {})
            fold_config_for_model["dataset"]["segments"] = fold_segments
            fold_task = build_task(fold_config_for_model)
            fold_model = init_instance_by_config(fold_task["model"])
            fold_model.fit(fold_dataset)

            # 评估该折（复用 ic_stability.evaluate_fold，含 Newey-West 显著性）
            eval_metrics = evaluate_fold(fold_model, fold_dataset)
            fold_result = {
                "fold": w["fold"],
                "train_start": w["train"][0],
                "train_end": w["train"][1],
                "test_start": w["test"][0],
                "test_end": w["test"][1],
                **eval_metrics,
            }
            fold_ics.append(fold_result)

            logger.debug("Fold %d IC_mean=%.4f",
                         w["fold"], eval_metrics.get("ic_mean") or 0)

        except Exception as e:
            logger.warning("rolling_cross_validation Fold %d 失败: %s", w["fold"], e)
            fold_ics.append({
                "fold": w["fold"],
                "ic_mean": None,
                "error": str(e),
            })

    n_success = len([f for f in fold_ics if f.get("ic_mean") is not None])
    logger.debug("rolling_cross_validation 完成: %d/%d folds 成功", n_success, len(windows))

    return {"fold_ics": fold_ics}