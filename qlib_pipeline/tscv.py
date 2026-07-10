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

import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from qlib_pipeline.numpy_compat import *  # noqa

logger = logging.getLogger(__name__)


class _PurgedDatasetH:
    """支持离散日期列表的 DatasetH 替代实现。

    问题背景：
      Qlib DatasetH 的 segments 语义是连续时间区间 [start, end]，
      当 segments["train"] = [train_dates[0], train_dates[-1]] 时，
      该区间会包含被 purge/embargo 特意排除的测试窗口本身，
      导致训练数据泄露到测试期。

    解决方案：
      本类接收已 fit 的 handler 和离散日期列表，
      fetch 全量 DataFrame 后按日期 isin() 过滤，
      确保只有 train_dates 中出现的日期才进入训练集。

    接口兼容性：
      提供 prepare(segment, col_set, data_key) 方法，
      与 Qlib DatasetH 的接口一致，LGBModel.fit/predict 和
      ic_stability.evaluate_fold 可直接使用。
    """

    def __init__(self, handler, segments: dict):
        """
        Args:
            handler: 已 fit 的 Qlib DataHandler 实例
            segments: {"train": [date1, date2, ...], "test": [date3, date4, ...]}
                      值为离散日期字符串列表（非 [start, end] 连续区间）
        """
        self.handler = handler
        self._segments = segments
        # 预取全量数据（handler.fetch 是内存操作，不重复 IO）
        self._full_cache: Dict[tuple, pd.DataFrame] = {}

    @property
    def segments(self):
        return self._segments

    def _get_dates(self, segment: str) -> List[str]:
        slc = self._segments.get(segment, [])
        if isinstance(slc, (list, tuple)):
            return list(slc)
        return [slc] if slc else []

    def prepare(self, segments, col_set="all", data_key=None, **kwargs):
        """与 Qlib DatasetH.prepare 接口兼容。

        Parameters
        ----------
        segments : str | list[str]
            段名如 "train"、"test"，或它们的列表
        col_set : str
            列组名: "all" | "feature" | "label" | ["feature", "label"]
        data_key : str
            Qlib DataHandlerLP.DK_I / DK_L / DK_R，传递给 handler.fetch
        """
        from qlib.data.dataset.handler import DataHandlerLP

        if data_key is None:
            data_key = DataHandlerLP.DK_I

        # 支持多段请求 ["train", "test"]
        if isinstance(segments, (list, tuple)) and not isinstance(segments, str):
            return [self._prepare_single(seg, col_set, data_key) for seg in segments]
        return self._prepare_single(segments, col_set, data_key)

    def _prepare_single(self, segment: str, col_set, data_key) -> pd.DataFrame:
        dates = self._get_dates(segment)
        if not dates:
            return pd.DataFrame()

        # 缓存键: (col_set_key, data_key)
        if isinstance(col_set, list):
            col_set_key = tuple(sorted(col_set))
        else:
            col_set_key = col_set
        cache_key = (col_set_key, data_key)

        if cache_key not in self._full_cache:
            self._full_cache[cache_key] = self.handler.fetch(col_set=col_set, data_key=data_key)

        full_df = self._full_cache[cache_key]
        date_index = pd.to_datetime(dates)
        mask = full_df.index.get_level_values(0).isin(date_index)
        return full_df.loc[mask]


def _resolve_label_horizon(config: dict) -> int:
    """从 config 的 labels.primary 解析标签前瞻天数。

    支持的 label 名称格式: ret_Nd / xs_ret_Nd / alpha_Nd / up_down_Nd
    例如 "xs_ret_60d" → 60

    Args:
        config: workflow config dict

    Returns:
        标签前瞻天数，解析失败时回退到默认值 20
    """
    import re
    primary = config.get("labels", {}).get("primary", "")
    if primary:
        m = re.match(r"^(ret|xs_ret|alpha|up_down)_(\d+)d$", primary)
        if m:
            return int(m.group(2))
        logger.warning("无法从 labels.primary='%s' 解析 horizon，使用默认 20", primary)
    return 20


def _resolve_purge_embargo(
    purge_days: Optional[int],
    embargo_days: Optional[int],
    label_horizon: int,
) -> Tuple[int, int]:
    """根据 label_horizon 动态计算 purge_days 和 embargo_days 的默认值。

    Lopez de Prado 建议:
      - purge_days 应覆盖标签窗口重叠，默认 = label_horizon
      - embargo_days 应为正值，默认 = max(1, label_horizon // 2)

    Args:
        purge_days: 用户指定的 purge_days，None 表示用默认值
        embargo_days: 用户指定的 embargo_days，None 表示用默认值
        label_horizon: 标签前瞻天数

    Returns:
        (purge_days, embargo_days) 解析后的最终值
    """
    if purge_days is None:
        purge_days = label_horizon
    if embargo_days is None:
        embargo_days = max(1, label_horizon // 2)
    return purge_days, embargo_days


def generate_purged_kfold_splits(
    dates: List[str],
    n_splits: int = 5,
    purge_days: Optional[int] = None,
    embargo_days: Optional[int] = None,
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
        purge_days: 标签窗口重叠的净化天数（None 时自动设为 label_horizon）
        embargo_days: 测试集后的禁运天数（None 时自动设为 max(1, label_horizon // 2)）
        label_horizon: 标签的前瞻天数，用于计算标签窗口重叠

    Yields:
        (train_dates, test_dates) for each fold
    """
    purge_days, embargo_days = _resolve_purge_embargo(
        purge_days, embargo_days, label_horizon
    )
    n = len(dates)
    date_dt = pd.to_datetime(dates)
    # 交易日 → 日历日折算（×7/5）：60 交易日 ≈ 84 日历日
    #   确保 purge/embargo/label 窗口完全覆盖标签前瞻期
    purge_td = pd.Timedelta(days=int(purge_days * 7 / 5))
    embargo_td = pd.Timedelta(days=int(embargo_days * 7 / 5))
    label_td = pd.Timedelta(days=int(label_horizon * 7 / 5))

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
        purge_days: 标签窗口重叠净化天数（None 时自动 = label_horizon）
        embargo_days: 测试集后禁运天数（None 时自动 = max(1, label_horizon // 2)）
        label_horizon: 标签前瞻天数（None 时从 config.labels.primary 解析）
        min_train_days: minimum training days required
    """

    def __init__(self, config: dict, n_splits: int = 5,
                 purge_days: Optional[int] = None, embargo_days: Optional[int] = None,
                 label_horizon: Optional[int] = None,
                 min_train_days: int = 252):
        self.config = config
        self.n_splits = n_splits
        # 从 config 解析 label_horizon（若未显式传入）
        if label_horizon is None:
            label_horizon = _resolve_label_horizon(config)
        self.label_horizon = label_horizon
        # 动态计算 purge_days / embargo_days 默认值
        self.purge_days, self.embargo_days = _resolve_purge_embargo(
            purge_days, embargo_days, self.label_horizon
        )
        self.min_train_days = min_train_days
        self.fold_results: List[Dict] = []

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

    def _load_calendar(self) -> List[str]:
        """从 Qlib 数据加载交易日历，时间范围从 config 读取（不再硬编码）。"""
        from qlib.data import D
        dh = self.config.get("data_handler", {})
        start = dh.get("start_time", "2020-01-01")
        end = dh.get("end_time", "2025-12-31")
        try:
            cal = D.calendar(start_time=start, end_time=end)
            return [str(c) for c in cal]
        except Exception:
            return pd.date_range(start, end, freq="B").strftime("%Y-%m-%d").tolist()

    def run(self) -> pd.DataFrame:
        """
        Execute Purged K-Fold TSCV.

        关键修复（2026-07-10）：
          原实现将离散 train_dates 喂给 Qlib DatasetH 的 segments 时只取
          [train_dates[0], train_dates[-1]]，Qlib 会按连续区间取数，
          导致测试期数据原封不动地混入训练集。
          现改用 _PurgedDatasetH，按离散日期 isin() 过滤，
          确保只有 train_dates 中出现的日期才进入训练集。

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

        # 通过 build_task 构建 task dict，确保 label 表达式、CSMedianSubtract、
        # RankLGBModel 路由等逻辑与主管线完全一致
        from qlib_pipeline.train import build_task
        import copy as _copy

        for i, (train_dates, test_dates) in enumerate(splits):
            if len(train_dates) < self.min_train_days:
                logger.warning("Fold %d: 跳过(训练天数=%d < %d)", i, len(train_dates), self.min_train_days)
                continue

            logger.info("Fold %d: train=%s~%s (%d days), test=%s~%s (%d days)",
                         i, train_dates[0], train_dates[-1], len(train_dates),
                         test_dates[0], test_dates[-1], len(test_dates))

            # 深拷贝原 config，只更新时间字段，保留 label/processors/freq/qlib_lgb 等
            fold_config = _copy.deepcopy(self.config)
            fold_config.setdefault("data_handler", {})

            # handler 的数据范围必须覆盖所有 train+test 日期
            # （双向净化时 train_dates 可能跨越 test 段前后）
            all_dates = sorted(set(train_dates + test_dates))
            fold_config["data_handler"]["start_time"] = all_dates[0]
            fold_config["data_handler"]["end_time"] = all_dates[-1]

            # processor 拟合范围：只用测试期之前的训练数据，避免标准化器
            # 统计量泄露测试期信息
            test_start = test_dates[0]
            pre_test_train = [d for d in train_dates if d < test_start]
            fit_end = pre_test_train[-1] if pre_test_train else train_dates[-1]
            fold_config["data_handler"]["fit_start_time"] = train_dates[0]
            fold_config["data_handler"]["fit_end_time"] = fit_end

            fold_config.setdefault("dataset", {})
            fold_config["dataset"]["handler"] = handler
            # segments 保留为占位（build_task 需要它），实际由 _PurgedDatasetH 覆盖
            fold_config["dataset"]["segments"] = {
                "train": [train_dates[0], train_dates[-1]],
                "test": [test_dates[0], test_dates[-1]],
            }

            # 通过 build_task 构建 task dict（处理 label/CSMedianSubtract/RankLGBModel）
            task = build_task(fold_config)

            try:
                # 先用 init_instance_by_config 创建并 fit handler
                handler_instance = init_instance_by_config(task["dataset"]["kwargs"]["handler"])

                # 用离散日期列表构建 _PurgedDatasetH，替代标准 DatasetH
                # 这是本次修复的核心：确保训练数据只包含 train_dates 中的日期
                dataset = _PurgedDatasetH(
                    handler=handler_instance,
                    segments={
                        "train": train_dates,
                        "test": test_dates,
                    },
                )

                model = init_instance_by_config(task["model"])

                with R.start(experiment_name=f"tscv_fold_{i}"):
                    R.log_params(**flatten_dict(task))
                    R.log_params(handler_type=handler)
                    R.log_params(
                        purge_strategy="discrete_date_isin",
                        n_train_dates=len(train_dates),
                        n_test_dates=len(test_dates),
                        fit_end_time=fit_end,
                    )
                    model.fit(dataset)
                    rid = R.get_recorder().id

                # 评估：在测试集上计算 IC 指标
                from qlib_pipeline.ic_stability import evaluate_fold, get_feature_importance
                eval_metrics = evaluate_fold(model, dataset, label_horizon=self.label_horizon)
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


def run_tscv(config: dict, n_splits: int = 5,
             purge_days: Optional[int] = None, embargo_days: Optional[int] = None,
             label_horizon: Optional[int] = None,
             output_dir: str = "output/tscv") -> pd.DataFrame:
    """Convenience wrapper for TSCV.

    purge_days/embargo_days/label_horizon 默认 None 时自动从 config 解析：
      - label_horizon 从 config.labels.primary 提取（如 "xs_ret_60d" → 60）
      - purge_days 默认 = label_horizon
      - embargo_days 默认 = max(1, label_horizon // 2)
    """
    cv = PurgedKFoldCV(config, n_splits=n_splits, purge_days=purge_days,
                       embargo_days=embargo_days, label_horizon=label_horizon)
    df = cv.run()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path / "tscv_results.csv", index=False)
    logger.info("TSCV 结果已保存到: %s", output_path / "tscv_results.csv")

    return df