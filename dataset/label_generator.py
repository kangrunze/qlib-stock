"""
标签（目标变量）生成器

生成未来收益率标签和超额收益（Alpha）标签。
核心原则：标签仅使用未来（t+1之后）数据，避免前视偏差。
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

# ---- 配置 ----
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

_CONFIG_PATH = _PROJECT_ROOT / "config" / "settings.yaml"
with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    _config = yaml.safe_load(f)

_labels_conf = _config.get("labels", {})
_paths = _config.get("paths", {})

INDEX_DIR = Path(_paths.get("index_dir", "data/index"))
PARQUET_DIR = Path(_paths.get("parquet_dir", "data/daily"))

# 标签类型配置
LABEL_TYPES = _labels_conf.get("types", [])
PRIMARY_LABEL = _labels_conf.get("primary", "alpha_20d")


class LabelGenerator:
    """
    标签生成器

    基于日线行情数据生成未来收益率标签和Alpha（超额收益）标签。
    所有标签仅使用 future-only 数据，严格避免前视偏差。

    Attributes:
        benchmark_code: Alpha基准指数代码，默认中证500 (000905)
    """

    def __init__(self, benchmark_code: str = "000905"):
        self.benchmark_code = benchmark_code
        self._benchmark_data: Optional[pd.DataFrame] = None

    def _load_benchmark(self) -> pd.DataFrame:
        """加载基准指数数据"""
        if self._benchmark_data is not None:
            return self._benchmark_data

        index_path = INDEX_DIR / f"{self.benchmark_code}.parquet"
        if index_path.exists():
            df = pd.read_parquet(str(index_path))
            df["date"] = pd.to_datetime(df["date"])
            self._benchmark_data = df.set_index("date").sort_index()
            logger.info(
                "加载基准指数 %s: %d 条数据, %s ~ %s",
                self.benchmark_code, len(df),
                df["date"].min().strftime("%Y-%m-%d"),
                df["date"].max().strftime("%Y-%m-%d"),
            )
        else:
            logger.warning("基准指数文件不存在: %s", index_path)
            self._benchmark_data = pd.DataFrame()

        return self._benchmark_data

    def generate_labels(
        self,
        df: pd.DataFrame,
        label_type: str = "alpha_20d",
    ) -> pd.DataFrame:
        """
        为单只股票生成标签

        Args:
            df: 单只股票日线DataFrame，必须包含 date, close 列
            label_type: 标签类型:
                - ret_20d: 未来20日收益率
                - ret_60d: 未来60日收益率
                - alpha_20d: 20日超额收益（vs基准指数）
                - alpha_60d: 60日超额收益（vs基准指数）

        Returns:
            包含 date, label 列的DataFrame
        """
        # 解析horizon
        label_configs = {lt["name"]: lt for lt in LABEL_TYPES}
        if label_type not in label_configs:
            raise ValueError(
                f"不支持的标签类型: {label_type}，支持: {list(label_configs.keys())}"
            )

        cfg = label_configs[label_type]
        horizon = cfg.get("horizon", 20)
        benchmark = cfg.get("benchmark", None)

        # 数据准备
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        if len(df) < horizon + 1:
            logger.warning("数据不足: %s, len=%d < horizon=%d+1", label_type, len(df), horizon)
            return pd.DataFrame(columns=["date", "label"])

        # 计算未来收益率（t+1 到 t+horizon）
        df["close_future"] = df["close"].shift(-horizon)
        df["ret"] = (df["close_future"] - df["close"]) / df["close"]

        # 如果标签是Alpha类型，需要计算超额收益
        if benchmark and label_type.startswith("alpha_"):
            benchmark_data = self._load_benchmark()
            if benchmark_data.empty:
                logger.warning("无法加载基准指数数据，回退为纯收益率标签")
                df["label"] = df["ret"]
            else:
                df = self._compute_alpha(df, benchmark_data, horizon)
        else:
            df["label"] = df["ret"]

        # 去掉最后 horizon 天（没有未来数据）
        result = df[["date", "label"]].dropna(subset=["label"]).copy()

        return result

    def _compute_alpha(
        self,
        df: pd.DataFrame,
        benchmark: pd.DataFrame,
        horizon: int,
    ) -> pd.DataFrame:
        """
        计算超额收益 Alpha

        Args:
            df: 股票DataFrame (含 date, close, ret)
            benchmark: 基准指数DataFrame (index=date, 含close)
            horizon: 预测期限

        Returns:
            df with 'label' column added (Alpha)
        """
        # 为股票数据匹配基准指数收益率
        bench_returns = benchmark["close"].pct_change(horizon).shift(-horizon)
        bench_returns = bench_returns.rename("bench_ret")

        # 将日期转换为字符串用于匹配
        df = df.copy()
        df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

        bench_returns = bench_returns.reset_index()
        bench_returns["date_str"] = bench_returns["date"].dt.strftime("%Y-%m-%d")

        # 合并基准收益率
        df = df.merge(
            bench_returns[["date_str", "bench_ret"]],
            on="date_str",
            how="left",
        )

        # Alpha = 股票收益率 - 基准收益率
        df["label"] = df["ret"] - df["bench_ret"]

        # 清理辅助列
        df = df.drop(columns=["date_str", "bench_ret"], errors="ignore")

        return df

    def generate_all_labels(
        self,
        data_dict: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.DataFrame]:
        """
        为多只股票批量生成所有标签类型

        Args:
            data_dict: {code: DataFrame} 股票数据字典

        Returns:
            {code: DataFrame with label columns}
        """
        if not LABEL_TYPES:
            logger.warning("配置中未定义标签类型，使用默认标签")
            _default_types = [
                {"name": "alpha_20d", "horizon": 20, "benchmark": "000905"}
            ]
        else:
            _default_types = LABEL_TYPES

        results = {}
        for code, df in data_dict.items():
            try:
                label_df = df[["date", "code"]].copy() if "code" in df.columns else df[["date"]].copy()
                label_df["code"] = label_df.get("code", code)

                for label_cfg in _default_types:
                    label_type = label_cfg["name"]
                    labels = self.generate_labels(df, label_type)
                    if labels.empty:
                        continue
                    labels = labels.rename(columns={"label": f"label_{label_type}"})
                    # 统一日期类型，避免 object 与 datetime64 不匹配
                    label_df["date"] = pd.to_datetime(label_df["date"])
                    labels["date"] = pd.to_datetime(labels["date"])
                    label_df = label_df.merge(labels, on="date", how="left")

                # 只保留有至少一个标签的行
                label_cols = [c for c in label_df.columns if c.startswith("label_")]
                if label_cols:
                    label_df = label_df.dropna(subset=label_cols, how="all")
                    results[code] = label_df

            except Exception as e:
                logger.error("code=%s 标签生成失败: %s", code, str(e)[:120])

        logger.info("批量标签生成完成: %d 只股票", len(results))
        return results

    def validate_date_alignment(
        self,
        df: pd.DataFrame,
        label_type: str = "alpha_20d",
    ) -> bool:
        """
        验证标签日期对齐: 确保所有标签使用未来数据

        Args:
            df: 已生成标签的DataFrame
            label_type: 标签类型名称

        Returns:
            是否通过验证
        """
        label_cfg = {lt["name"]: lt for lt in LABEL_TYPES}.get(label_type, {})
        horizon = label_cfg.get("horizon", 20)

        if len(df) < horizon + 1:
            return True

        # 确保最后 horizon 天没有被标记
        df = df.sort_values("date")
        last_horizon = df["date"].iloc[-horizon:]

        col_name = f"label_{label_type}"
        if col_name in df.columns:
            last_labels = df.iloc[-horizon:][col_name]
            if last_labels.notna().any():
                logger.warning(
                    "日期对齐检查失败: 最后%d天仍有非空标签值", horizon
                )
                return False

        return True


def generate_labels(
    df: pd.DataFrame,
    label_type: str = "alpha_20d",
    benchmark_code: str = "000905",
) -> pd.DataFrame:
    """便捷标签生成函数"""
    gen = LabelGenerator(benchmark_code=benchmark_code)
    return gen.generate_labels(df, label_type)


def generate_all_labels(
    data_dict: Dict[str, pd.DataFrame],
    benchmark_code: str = "000905",
) -> Dict[str, pd.DataFrame]:
    """便捷批量标签生成函数"""
    gen = LabelGenerator(benchmark_code=benchmark_code)
    return gen.generate_all_labels(data_dict)