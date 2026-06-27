"""
数据质量校验器

校验Parquet文件中股票数据的完整性、一致性、异常值和重复值。
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ---- 加载配置 ----
_CONFIG_PATH = _PROJECT_ROOT / "config" / "settings.yaml"
with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    _config = yaml.safe_load(f)

_paths = _config.get("paths", {})
PARQUET_DIR = Path(_paths.get("parquet_dir", "data/daily"))
REFERENCE_DIR = Path(_paths.get("reference_dir", "data/reference"))

# 必须存在的列
REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume", "code"]

# 价格跳变阈值（百分比）
PRICE_JUMP_THRESHOLD = 0.20  # 单日涨跌超过20%视为异常
# 零量但价格变化的容忍度
ZERO_VOL_PRICE_CHANGE_TOLERANCE = 0.001


class DataValidator:
    """
    数据质量校验器

    对Parquet股票数据执行以下检查：
    - 完整性检查（缺失日期、无数据股票）
    - 一致性检查（价格关系：low<=high, open/close在low~high之间）
    - 异常检查（极端价格跳变、零量价格变化）
    - 重复检查

    Attributes:
        parquet_dir: 日线数据Parquet目录
        reference_dir: 参考数据目录
    """

    def __init__(
        self,
        parquet_dir: Optional[str] = None,
        reference_dir: Optional[str] = None,
    ):
        self.parquet_dir = Path(parquet_dir) if parquet_dir else PARQUET_DIR
        self.reference_dir = Path(reference_dir) if reference_dir else REFERENCE_DIR
        self._issues: Dict[str, List[Dict]] = {}
        self._stats: Dict[str, dict] = {}

    def validate_file(self, filepath: str) -> Dict:
        """
        校验单个Parquet文件

        Args:
            filepath: Parquet文件路径

        Returns:
            校验结果字典:
            {
                "code": str,
                "rows": int,
                "date_range": (min, max),
                "issues": List[Dict],
                "passed": bool
            }
        """
        filepath = Path(filepath)
        code = filepath.stem
        result = {
            "code": code,
            "rows": 0,
            "date_range": (None, None),
            "issues": [],
            "passed": True,
        }

        try:
            df = pd.read_parquet(str(filepath))
            result["rows"] = len(df)

            if len(df) == 0:
                result["issues"].append({
                    "type": "empty",
                    "severity": "error",
                    "message": "文件为空"
                })
                result["passed"] = False
                return result

            # ---- 列完整性检查 ----
            self._check_columns(df, result)

            # ---- 日期检查 ----
            self._check_dates(df, result)

            # ---- 价格一致性检查 ----
            self._check_price_consistency(df, result)

            # ---- 异常检查 ----
            self._check_anomalies(df, result)

            # ---- 重复检查 ----
            self._check_duplicates(df, result)

            result["passed"] = len(result["issues"]) == 0

        except Exception as e:
            result["issues"].append({
                "type": "read_error",
                "severity": "error",
                "message": f"读取文件失败: {str(e)[:200]}"
            })
            result["passed"] = False

        return result

    def _check_columns(self, df: pd.DataFrame, result: Dict) -> None:
        """检查必需列是否存在"""
        missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            result["issues"].append({
                "type": "missing_columns",
                "severity": "error",
                "message": f"缺少列: {missing_cols}"
            })

    def _check_dates(self, df: pd.DataFrame, result: Dict) -> None:
        """检查日期列"""
        if "date" not in df.columns:
            result["issues"].append({
                "type": "missing_date",
                "severity": "error",
                "message": "缺少date列"
            })
            return

        # 确保date是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            try:
                df["date"] = pd.to_datetime(df["date"])
            except Exception:
                result["issues"].append({
                    "type": "invalid_date",
                    "severity": "error",
                    "message": "date列无法转换为datetime"
                })
                return

        result["date_range"] = (
            df["date"].min().strftime("%Y-%m-%d"),
            df["date"].max().strftime("%Y-%m-%d"),
        )

        # 检查日期重复
        dup_dates = df["date"].duplicated().sum()
        if dup_dates > 0:
            result["issues"].append({
                "type": "duplicate_dates",
                "severity": "warning",
                "message": f"有 {dup_dates} 个重复日期"
            })

    def _check_price_consistency(self, df: pd.DataFrame, result: Dict) -> None:
        """检查价格关系一致性"""
        price_cols = {"open", "high", "low", "close"}
        if not price_cols.issubset(set(df.columns)):
            return

        # low <= high
        bad_hl = df[df["low"] > df["high"]]
        if len(bad_hl) > 0:
            result["issues"].append({
                "type": "low_gt_high",
                "severity": "error",
                "message": f"low > high 有 {len(bad_hl)} 行",
                "sample_dates": bad_hl["date"].head(5).dt.strftime("%Y-%m-%d").tolist()
            })

        # open 应在 low 和 high 之间
        bad_open = df[(df["open"] < df["low"]) | (df["open"] > df["high"])]
        if len(bad_open) > 0:
            result["issues"].append({
                "type": "open_out_of_range",
                "severity": "error",
                "message": f"open不在[low,high]范围内: {len(bad_open)}行"
            })

        # close 应在 low 和 high 之间
        bad_close = df[(df["close"] < df["low"]) | (df["close"] > df["high"])]
        if len(bad_close) > 0:
            result["issues"].append({
                "type": "close_out_of_range",
                "severity": "error",
                "message": f"close不在[low,high]范围内: {len(bad_close)}行"
            })

    def _check_anomalies(self, df: pd.DataFrame, result: Dict) -> None:
        """检查异常值"""
        if "close" not in df.columns:
            return

        # 极端价格跳变
        if len(df) > 1:
            pct_changes = df["close"].pct_change().abs()
            extreme_jumps = pct_changes[pct_changes > PRICE_JUMP_THRESHOLD]
            if len(extreme_jumps) > 0:
                jump_idx = extreme_jumps.index.tolist()
                result["issues"].append({
                    "type": "extreme_price_jump",
                    "severity": "warning",
                    "message": f"价格跳变>{PRICE_JUMP_THRESHOLD*100:.0f}%: {len(extreme_jumps)}处",
                    "sample_dates": [
                        df.loc[i, "date"].strftime("%Y-%m-%d") if hasattr(df.loc[i, "date"], "strftime")
                        else str(df.loc[i, "date"])
                        for i in jump_idx[:5]
                    ]
                })

        # 零量但价格变化
        if "volume" in df.columns:
            zero_vol = df["volume"] == 0
            if zero_vol.any():
                # 有量的日子的价格变化
                price_chg = df["close"].diff().abs()
                zero_vol_with_chg = zero_vol & (price_chg > ZERO_VOL_PRICE_CHANGE_TOLERANCE)
                if zero_vol_with_chg.any():
                    result["issues"].append({
                        "type": "zero_vol_price_change",
                        "severity": "warning",
                        "message": f"零成交量但价格变化: {zero_vol_with_chg.sum()}处"
                    })

    def _check_duplicates(self, df: pd.DataFrame, result: Dict) -> None:
        """检查重复行"""
        if "date" not in df.columns:
            return

        dup_mask = df.duplicated(subset=["date"], keep=False)
        if dup_mask.any():
            result["issues"].append({
                "type": "duplicate_rows",
                "severity": "error",
                "message": f"完全重复行: {dup_mask.sum()}行"
            })

    def validate_all(self) -> Dict[str, Dict]:
        """
        校验parquet_dir下所有Parquet文件

        Returns:
            Dict[code, validation_result]
        """
        all_results = {}
        parquet_files = sorted(self.parquet_dir.glob("*.parquet"))

        if not parquet_files:
            logger.warning("未找到Parquet文件: %s", self.parquet_dir)
            return all_results

        logger.info("开始校验 %d 个Parquet文件...", len(parquet_files))

        for i, fp in enumerate(parquet_files):
            if (i + 1) % 100 == 0:
                logger.info("  校验进度: %d/%d", i + 1, len(parquet_files))
            result = self.validate_file(str(fp))
            all_results[result["code"]] = result

        logger.info("校验完成: %d 个文件", len(all_results))
        return all_results

    def generate_report(self, results: Optional[Dict[str, Dict]] = None) -> str:
        """
        生成校验报告

        Args:
            results: validate_all()的返回值, 如果为None则自动执行

        Returns:
            报告文本
        """
        if results is None:
            results = self.validate_all()

        total = len(results)
        passed = sum(1 for r in results.values() if r["passed"])
        failed = total - passed

        # 统计各类问题
        issue_counts: Dict[str, int] = {}
        error_count = 0
        warning_count = 0

        for r in results.values():
            for issue in r["issues"]:
                itype = issue["type"]
                issue_counts[itype] = issue_counts.get(itype, 0) + 1
                if issue["severity"] == "error":
                    error_count += 1
                elif issue["severity"] == "warning":
                    warning_count += 1

        # 构建报告
        lines = []
        lines.append("=" * 70)
        lines.append("  数据质量校验报告")
        lines.append(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"  总文件数: {total}")
        lines.append(f"  通过: {passed} ({100*passed/total:.1f}%)" if total > 0 else "  通过: 0")
        lines.append(f"  未通过: {failed} ({100*failed/total:.1f}%)" if total > 0 else "  未通过: 0")
        lines.append(f"  错误数: {error_count}")
        lines.append(f"  警告数: {warning_count}")
        lines.append("")

        if issue_counts:
            lines.append("-" * 70)
            lines.append("  问题类型统计:")
            lines.append("-" * 70)
            for itype, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
                lines.append(f"    {itype}: {count}")
            lines.append("")

        if failed > 0:
            lines.append("-" * 70)
            lines.append("  未通过文件详情 (前20个):")
            lines.append("-" * 70)
            failed_files = [(code, r) for code, r in results.items() if not r["passed"]]
            for code, r in failed_files[:20]:
                lines.append(f"  [{code}] rows={r['rows']}")
                for issue in r["issues"]:
                    severity_mark = "[ERROR]" if issue["severity"] == "error" else "[WARN]"
                    lines.append(f"    {severity_mark} {issue['type']}: {issue['message']}")

        lines.append("")
        lines.append("=" * 70)
        lines.append("  报告结束")
        lines.append("=" * 70)

        report = "\n".join(lines)
        logger.info(report)
        return report


def validate_file(filepath: str) -> Dict:
    """便捷校验单个文件"""
    validator = DataValidator()
    return validator.validate_file(filepath)


def validate_all() -> Dict[str, Dict]:
    """便捷校验所有文件"""
    validator = DataValidator()
    return validator.validate_all()


def generate_report() -> str:
    """便捷生成校验报告"""
    validator = DataValidator()
    return validator.generate_report()