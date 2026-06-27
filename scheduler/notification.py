"""
通知/日志模块 - 简化的Pipeline事件日志记录

功能:
    - 带有时间戳的日志记录
    - 错误日志（含traceback）
    - Pipeline运行摘要
    - 日志文件轮转管理

使用方法:
    notifier = Notification(log_dir="logs")
    notifier.send_log("Pipeline started", level="INFO")
    notifier.send_error("Data loading failed", exception)
    notifier.send_summary(pipeline_result)
"""

import logging
import logging.handlers
import traceback
import json
import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# 设置模块日志
logger = logging.getLogger(__name__)


class Notification:
    """Pipeline通知和日志管理器。

    记录pipeline运行中的各类事件，支持文件和控制台两种输出。

    Attributes:
        log_dir: 日志目录。
        log_file: 日志文件路径。
        logger: Python logger实例。
    """

    def __init__(
        self,
        log_dir: str = "logs",
        log_file: str = "pipeline.log",
        console_output: bool = True,
        max_log_size_mb: int = 50,
        backup_count: int = 10,
    ):
        """初始化通知管理器。

        Args:
            log_dir: 日志文件目录。
            log_file: 日志文件名。
            console_output: 是否同时输出到控制台。
            max_log_size_mb: 单个日志文件最大大小（MB）。
            backup_count: 保留的日志备份数。
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / log_file

        # 创建专用logger
        self._logger = logging.getLogger("pipeline_notification")
        self._logger.setLevel(logging.INFO)
        self._logger.handlers.clear()

        # 文件handler（带轮转）
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_file,
            maxBytes=max_log_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        self._logger.addHandler(file_handler)

        # 控制台handler
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            self._logger.addHandler(console_handler)

    def send_log(self, message: str, level: str = "INFO") -> None:
        """记录一条日志消息。

        Args:
            message: 日志消息。
            level: 日志级别 (DEBUG/INFO/WARNING/ERROR/CRITICAL)。
        """
        log_method = getattr(self._logger, level.lower(), self._logger.info)
        log_method(message)

    def send_error(
        self,
        message: str,
        exception: Optional[Exception] = None,
    ) -> None:
        """记录错误信息，附带traceback。

        Args:
            message: 错误描述。
            exception: 异常对象（可选）。
        """
        if exception:
            tb = "".join(traceback.format_exception(
                type(exception), exception, exception.__traceback__
            ))
            self._logger.error(f"{message}\n{tb}")
        else:
            self._logger.error(message)

    def send_warning(self, message: str) -> None:
        """记录警告信息。

        Args:
            message: 警告消息。
        """
        self._logger.warning(message)

    def send_summary(self, pipeline_result: Dict[str, Any]) -> None:
        """输出pipeline运行摘要。

        Args:
            pipeline_result: pipeline运行结果字典。
        """
        separator = "=" * 70
        self._logger.info(separator)
        self._logger.info("PIPELINE 运行摘要")
        self._logger.info(separator)

        # 运行状态
        status = pipeline_result.get("status", "UNKNOWN")
        self._logger.info(f"  状态: {status}")
        self._logger.info(f"  开始时间: {pipeline_result.get('start_time', 'N/A')}")
        self._logger.info(f"  结束时间: {pipeline_result.get('end_time', 'N/A')}")
        self._logger.info(f"  总耗时: {pipeline_result.get('duration_seconds', 'N/A')} 秒")

        # 各步骤详情
        steps = pipeline_result.get("steps", {})
        if steps:
            self._logger.info(f"\n  步骤明细:")
            for step_name, step_info in steps.items():
                status_icon = "OK" if step_info.get("status") == "success" else "FAIL"
                duration = step_info.get("duration_seconds", 0)
                self._logger.info(
                    f"    [{status_icon}] {step_name}: {duration:.1f}s"
                )
                if step_info.get("error"):
                    self._logger.info(f"      Error: {step_info['error']}")

        # 结果统计
        summary_stats = pipeline_result.get("summary", {})
        for key, value in summary_stats.items():
            self._logger.info(f"  {key}: {value}")

        self._logger.info(separator)

    def send_alert(self, message: str, alert_type: str = "WARNING") -> None:
        """发送高优先级告警。

        Args:
            message: 告警消息。
            alert_type: 告警类型。
        """
        self._logger.warning(f"[ALERT] [{alert_type}] {message}")

    def save_pipeline_log(
        self, pipeline_result: Dict[str, Any], filename: Optional[str] = None
    ) -> None:
        """将pipeline结果保存为JSON日志文件。

        Args:
            pipeline_result: pipeline结果字典。
            filename: 输出文件名（None则自动生成）。
        """
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pipeline_{timestamp}.json"

        output_path = self.log_dir / filename

        try:
            # 处理不可序列化对象
            def default_serializer(obj):
                if hasattr(obj, "__dict__"):
                    return str(obj)
                return str(obj)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(pipeline_result, f, ensure_ascii=False, indent=2, default=default_serializer)

            self._logger.info(f"Pipeline日志已保存: {output_path}")
        except Exception as e:
            self._logger.error(f"保存pipeline日志失败: {e}")


def create_pipeline_result(
    status: str = "running",
    start_time: Optional[str] = None,
) -> Dict[str, Any]:
    """创建pipeline运行结果字典。

    Args:
        status: 初始状态。
        start_time: 开始时间字符串。

    Returns:
        pipeline结果字典。
    """
    if start_time is None:
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "status": status,
        "start_time": start_time,
        "end_time": None,
        "duration_seconds": 0,
        "steps": {},
        "summary": {},
        "errors": [],
    }


def complete_pipeline_result(
    result: Dict[str, Any],
    status: str = "success",
    summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """完成pipeline结果字典。

    Args:
        result: pipeline结果字典。
        status: 最终状态。
        summary: 摘要信息。

    Returns:
        完成后的结果字典。
    """
    end_time = datetime.datetime.now()
    result["end_time"] = end_time.strftime("%Y-%m-%d %H:%M:%S")

    if result.get("start_time"):
        try:
            start_dt = datetime.datetime.strptime(
                result["start_time"], "%Y-%m-%d %H:%M:%S"
            )
            result["duration_seconds"] = round(
                (end_time - start_dt).total_seconds(), 1
            )
        except Exception:
            result["duration_seconds"] = 0

    result["status"] = status
    if summary:
        result["summary"] = summary

    return result