# -*- coding: utf-8 -*-
"""
实验追踪器 — experiment_tracker.py

实现第 2.2 节和第 8.2 节的设计要求：
  1. 探索/确认阶段协议：明确划分两个不相交的数据窗口
  2. 每次实验（无论好坏）都要记录在案
  3. 确认阶段只允许对最终指定的一组配置运行一次
  4. 追踪探索阶段总共尝试了多少组配置，用于多重检验校正

使用方式:
    from research.experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker(storage_dir="output/experiments")
    tracker.log_exploration(config, metrics, phase="exploration")
    tracker.log_confirmation(config, metrics, phase="confirmation")
    tracker.summary()  # 输出实验总结 + 校正后的显著性阈值
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """探索/确认阶段实验追踪器。

    维护一个 JSON 日志文件，记录每一次实验的配置、指标、阶段和测试数据段。
    提供确认阶段防重复检查、实验总数统计、多重检验校正阈值计算。

    Args:
        storage_dir: 实验日志存储目录
        exploration_window: 探索窗口日期范围 [start, end]
        confirmation_window: 确认窗口日期范围 [start, end]
        embargo_days: 禁运期天数（探索窗口结束 → 确认窗口开始之间的间隔）
    """

    def __init__(
        self,
        storage_dir: str = "output/experiments",
        exploration_window: Optional[Tuple[str, str]] = None,
        confirmation_window: Optional[Tuple[str, str]] = None,
        embargo_days: int = 60,
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.storage_dir / "experiment_log.jsonl"

        self.exploration_window = exploration_window
        self.confirmation_window = confirmation_window
        self.embargo_days = embargo_days

        # 加载已有记录
        self._records: List[Dict] = self._load_records()

    def _load_records(self) -> List[Dict]:
        """加载已有的实验记录"""
        records = []
        if self.log_file.exists():
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            records.append(json.loads(line))
            except Exception as e:
                logger.warning("加载实验记录失败: %s", e)
        return records

    def _save_record(self, record: Dict):
        """追加一条实验记录"""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._records.append(record)

    def log_experiment(
        self,
        config: Dict,
        metrics: Dict,
        phase: str = "exploration",
        test_data_range: Optional[Tuple[str, str]] = None,
        recorder_id: Optional[str] = None,
        description: str = "",
    ) -> Dict:
        """记录一次实验。

        Args:
            config: 实验配置（关键参数摘要，不含完整配置）
            metrics: 本次实验的核心指标（IC、ICIR、NW 显著性等）
            phase: 阶段 ("exploration" | "confirmation")
            test_data_range: 测试数据日期范围
            recorder_id: Qlib Recorder ID
            description: 实验描述

        Returns:
            record dict
        """
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "phase": phase,
            "description": description,
            "recorder_id": recorder_id,
            "test_data_range": test_data_range,
            "config_summary": self._summarize_config(config),
            "metrics": metrics,
        }
        self._save_record(record)
        logger.info(
            "实验记录: phase=%s, desc=%s, n_total=%d",
            phase, description, self.count_exploration(),
        )
        return record

    def _summarize_config(self, config: Dict) -> Dict:
        """提取配置的关键参数摘要"""
        keys_of_interest = [
            "handler", "loss", "primary", "topk", "n_drop",
            "learning_rate", "max_depth", "num_leaves",
            "colsample_bytree", "subsample",
        ]
        summary = {}
        # 从 workflow_config 中提取
        for key in keys_of_interest:
            # 检查 qlib_lgb kwargs
            if "qlib_lgb" in config and key in config["qlib_lgb"].get("kwargs", {}):
                summary[key] = config["qlib_lgb"]["kwargs"][key]
            # 检查 dataset handler
            elif "dataset" in config and key == "handler":
                summary[key] = config["dataset"].get("handler", "unknown")
            # 检查 labels primary
            elif "labels" in config and key == "primary":
                summary[key] = config["labels"].get("primary", "unknown")
            # 检查 backtest strategy
            elif "backtest" in config and key in ["topk", "n_drop"]:
                summary[key] = config["backtest"]["strategy"]["kwargs"].get(key, "unknown")
        return summary

    def count_exploration(self) -> int:
        """统计探索阶段实验总数"""
        return sum(1 for r in self._records if r.get("phase") == "exploration")

    def count_confirmation(self) -> int:
        """统计确认阶段实验总数"""
        return sum(1 for r in self._records if r.get("phase") == "confirmation")

    def check_confirmation_protocol(self) -> Tuple[bool, str]:
        """检查确认阶段协议是否被遵守。

        规则：
          1. 确认阶段只能运行一次（第 2.2 节）
          2. 确认阶段的测试数据不能与探索阶段重叠
          3. 确认阶段的结果无论好坏都要如实记录，不允许因为不满意就回头修改

        Returns:
            (ok, message)
        """
        conf_count = self.count_confirmation()
        if conf_count > 1:
            return False, (
                f"确认阶段协议违规: 已运行 {conf_count} 次确认阶段实验，"
                "确认阶段只允许运行一次（第 2.2 节）。"
                "之前的结果应如实记录，不允许因为不满意就回头修改探索阶段的选择再跑一次确认窗口。"
            )

        if conf_count == 1:
            # 检查确认窗口的测试数据是否与探索阶段有重叠
            conf_records = [r for r in self._records if r.get("phase") == "confirmation"]
            conf_test = conf_records[0].get("test_data_range")
            if conf_test and len(conf_test) == 2:
                import pandas as pd
                conf_interval = pd.Interval(
                    pd.Timestamp(conf_test[0]), pd.Timestamp(conf_test[1]),
                    closed="both",
                )
                for r in self._records:
                    if r.get("phase") == "exploration" and r.get("test_data_range"):
                        exp_range = r["test_data_range"]
                        if not exp_range or len(exp_range) != 2:
                            continue
                        exp_interval = pd.Interval(
                            pd.Timestamp(exp_range[0]), pd.Timestamp(exp_range[1]),
                            closed="both",
                        )
                        if conf_interval.overlaps(exp_interval):
                            return False, (
                                f"确认阶段协议违规: 确认窗口 {list(conf_test)} 与探索阶段实验 "
                                f"{r.get('description', r.get('recorder_id', '未知'))} 的测试窗口 "
                                f"{list(exp_range)} 存在重叠，违反训练/验证/测试纪律"
                            )

        return True, "确认阶段协议检查通过"

    def get_corrected_significance(self, alpha: float = 0.05, method: str = "fdr_bh") -> Dict:
        """计算多重检验校正后的显著性阈值。

        Args:
            alpha: 目标显著性水平
            method: 校正方法 ("bonferroni" | "fdr_bh")

        Returns:
            dict with corrected_threshold, n_exploration, n_confirmation
        """
        n_exp = self.count_exploration()
        n_conf = self.count_confirmation()

        from research.significance import multiple_testing_correction

        # 收集探索阶段所有实验的 NW p 值
        p_values = []
        for r in self._records:
            if r.get("phase") == "exploration":
                metrics = r.get("metrics", {})
                p = metrics.get("nw_p_value")
                if p is not None:
                    p_values.append(p)

        if p_values:
            corr = multiple_testing_correction(p_values, method=method, alpha=alpha)
        else:
            corr = {"corrected_threshold": alpha, "significant_indices": [], "adjusted_p_values": [], "n_tests": 0}

        return {
            "corrected_threshold": corr["corrected_threshold"],
            "n_exploration": n_exp,
            "n_confirmation": n_conf,
            "n_with_p_values": len(p_values),
            "method": method,
            "significant_indices": corr["significant_indices"],
        }

    def summary(self, alpha: float = 0.05) -> str:
        """生成实验追踪汇总报告。

        Returns:
            格式化的报告字符串
        """
        n_exp = self.count_exploration()
        n_conf = self.count_confirmation()
        n_total = len(self._records)

        corr = self.get_corrected_significance(alpha=alpha)

        lines = [
            "=" * 60,
            "实验追踪汇总报告",
            "=" * 60,
            f"  探索阶段实验数:     {n_exp}",
            f"  确认阶段实验数:     {n_conf}",
            f"  实验总数:           {n_total}",
            "",
            "多重检验校正:",
            f"  方法:               {corr['method']}",
            f"  原始显著性水平:     {alpha}",
            f"  校正后阈值:         {corr['corrected_threshold']:.6f}",
            f"  有 NW p 值的实验:   {corr['n_with_p_values']}",
            "",
        ]

        # 协议检查
        ok, msg = self.check_confirmation_protocol()
        if ok:
            lines.append("  探索/确认协议:    ✓ 通过")
        else:
            lines.append(f"  探索/确认协议:    ✗ {msg}")

        lines.append("")
        lines.append("  探索阶段实验摘要:")

        for i, r in enumerate(self._records):
            if r.get("phase") == "exploration":
                desc = r.get("description", f"实验 #{i+1}")
                metrics = r.get("metrics", {})
                ic_mean = metrics.get("ic_mean", "?")
                nw_significant = metrics.get("nw_significant", False)
                nw_t = metrics.get("nw_t_stat", "?")
                lines.append(
                    f"    [{i+1}] {desc}: IC={ic_mean}, "
                    f"|t_NW|={nw_t}, NW显著={'✓' if nw_significant else '✗'}"
                )

        lines.append("")
        if n_exp > 1:
            lines.append(
                f"  ⚠ 探索阶段共尝试了 {n_exp} 组配置，"
                f"最终报告结果时应使用校正后的显著性阈值 {corr['corrected_threshold']:.6f} "
                f"（而非原始 alpha={alpha}），否则存在以比较偏误。"
            )

        lines.append("=" * 60)
        return "\n".join(lines)

    def clear(self):
        """清空所有实验记录（慎用，通常只在全新研究开始时使用）"""
        self._records = []
        if self.log_file.exists():
            self.log_file.unlink()
        logger.warning("实验记录已清空")


def _global_tracker() -> ExperimentTracker:
    """获取全局 ExperimentTracker 单例"""
    return ExperimentTracker(storage_dir="output/experiments")