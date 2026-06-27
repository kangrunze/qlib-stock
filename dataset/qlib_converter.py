"""
Parquet 到 Qlib 格式转换器

将 Parquet 数据转换为 Qlib 兼容的 DatasetH 格式。
保持 qlib 为可选依赖，仅在使用到相关功能时才需要安装。
"""

import sys
import logging
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

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
_split_conf = _config.get("dataset_split", {})

PARQUET_DIR = Path(_paths.get("parquet_dir", "data/daily"))
FEATURES_DIR = Path(_paths.get("features_dir", "data/features"))
DATA_DIR = Path(_paths.get("data_dir", "data"))


class QlibConverter:
    """
    Parquet转Qlib格式转换器

    将面板数据（Parquet）转换为 Qlib DatasetH 可读取的格式。
    Qlib 数据格式要求:
    - 按股票分目录, 每个股票下有多个 .bin 文件（按日期命名）
    - 或使用 calendar/instruments/features 三元组

    Attributes:
        data_dir: 输出数据根目录
        parquet_dir: 输入Parquet目录
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        parquet_dir: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.parquet_dir = Path(parquet_dir) if parquet_dir else PARQUET_DIR
        self._qlib_available = self._check_qlib()

    @staticmethod
    def _check_qlib() -> bool:
        """检查 qlib 是否可用"""
        try:
            import qlib  # noqa: F401
            return True
        except ImportError:
            logger.warning("qlib 未安装，QlibConverter 部分功能将不可用")
            return False

    # ==================== 数据结构构建 ====================

    def load_parquet_to_dict(
        self,
        codes: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        从 Parquet 文件加载数据到字典

        Args:
            codes: 股票代码列表，None表示加载全部
            start: 起始日期
            end: 结束日期

        Returns:
            {code: DataFrame} 字典
        """
        data_dict = {}

        if codes is None:
            parquet_files = sorted(self.parquet_dir.glob("*.parquet"))
            codes = [f.stem for f in parquet_files]

        for code in codes:
            fp = self.parquet_dir / f"{code}.parquet"
            if not fp.exists():
                logger.debug("code=%s 的 Parquet 文件不存在", code)
                continue

            df = pd.read_parquet(str(fp))
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])

            if start:
                df = df[df["date"] >= start]
            if end:
                df = df[df["date"] <= end]

            if len(df) > 0:
                data_dict[code] = df

        logger.info("加载 %d 只股票数据 from %s", len(data_dict), self.parquet_dir)
        return data_dict

    def build_panel_data(
        self,
        data_dict: Dict[str, pd.DataFrame],
        features_df: Optional[pd.DataFrame] = None,
        labels_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        构建面板数据 (Panel DataFrame)

        将散列的股票数据合并为一个统一的面板DataFrame。

        Args:
            data_dict: {code: DataFrame} 股票数据
            features_df: 特征DataFrame (可选, 按code+date合并)
            labels_df: 标签DataFrame (可选, 按code+date合并)

        Returns:
            面板DataFrame, 列: code, date, feature_*, label_*
        """
        # 合并日线数据
        all_dfs = []
        for code, df in data_dict.items():
            if len(df) == 0:
                continue
            cols_to_keep = [c for c in ["code", "date", "open", "high", "low", "close", "volume", "amount", "pct_chg", "turnover_rate"] if c in df.columns]
            subset = df[cols_to_keep].copy()
            if "code" not in subset.columns:
                subset["code"] = code
            all_dfs.append(subset)

        if not all_dfs:
            logger.warning("无可用数据构建面板")
            return pd.DataFrame()

        panel = pd.concat(all_dfs, ignore_index=True)
        panel["date"] = pd.to_datetime(panel["date"])

        # 合并特征
        if features_df is not None and len(features_df) > 0:
            features_df = features_df.copy()
            features_df["date"] = pd.to_datetime(features_df["date"])
            panel = panel.merge(
                features_df,
                on=["code", "date"],
                how="left",
            )

        # 合并标签
        if labels_df is not None and len(labels_df) > 0:
            labels_df = labels_df.copy()
            labels_df["date"] = pd.to_datetime(labels_df["date"])
            panel = panel.merge(
                labels_df,
                on=["code", "date"],
                how="left",
            )

        panel = panel.sort_values(["code", "date"]).reset_index(drop=True)
        logger.info(
            "面板数据构建完成: %d 行, %d 列, %d 只股票",
            len(panel), len(panel.columns), panel["code"].nunique()
        )
        return panel

    # ==================== Qlib 格式转换 ====================

    def convert_to_qlib_bin(
        self,
        data_dict: Dict[str, pd.DataFrame],
        output_dir: Optional[str] = None,
    ) -> str:
        """
        将数据转换为 Qlib 二进制格式 (bin文件)

        Qlib 要求的 bin 格式目录结构:
        data_dir/
          features/
            <instrument>/
              <date>.bin     (每个文件包含该股票当天的特征数组)

        Args:
            data_dict: {code: DataFrame} 股票数据字典
            output_dir: 输出目录

        Returns:
            输出目录路径
        """
        if not self._qlib_available:
            raise ImportError("qlib 未安装，无法转换为 Qlib 格式。请安装: pip install qlib")

        import qlib
        from qlib.data.dataset.loader import QlibDataLoader
        from qlib.data import D

        output_dir = Path(output_dir) if output_dir else self.data_dir / "qlib_bin"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Qlib 数据格式
        features_dir = output_dir / "features"
        features_dir.mkdir(exist_ok=True)

        calendar: List[str] = []

        for code, df in data_dict.items():
            df = df.sort_values("date").copy()
            inst_dir = features_dir / code
            inst_dir.mkdir(parents=True, exist_ok=True)

            for _, row in df.iterrows():
                date_str = row["date"].strftime("%Y-%m-%d") if isinstance(row["date"], pd.Timestamp) else str(row["date"])
                calendar.append(date_str)

                # 提取OHLCV特征
                feat_arr = np.array([
                    row.get("open", np.nan),
                    row.get("high", np.nan),
                    row.get("low", np.nan),
                    row.get("close", np.nan),
                    row.get("volume", np.nan),
                    row.get("amount", np.nan),
                    row.get("pct_chg", np.nan),
                    row.get("turnover_rate", np.nan),
                ], dtype=np.float32)

                bin_path = inst_dir / f"{date_str.replace('-', '')}.bin"
                with open(bin_path, "wb") as fout:
                    fout.write(feat_arr.tobytes())

        # 保存去重日历
        calendar = sorted(set(calendar))
        with open(output_dir / "calendars.txt", "w") as f:
            f.write("\n".join(calendar))

        # 保存 instruments
        with open(output_dir / "instruments.txt", "w") as f:
            f.write("\n".join(sorted(data_dict.keys())))

        logger.info(
            "Qlib bin 格式转换完成: %d 只股票, %d 个交易日 -> %s",
            len(data_dict), len(calendar), output_dir
        )
        return str(output_dir)

    def convert_to_qlib_csv(
        self,
        data_dict: Dict[str, pd.DataFrame],
        output_dir: Optional[str] = None,
    ) -> str:
        """
        将数据转换为 Qlib CSV 格式

        每个股票一个daily.csv文件，放在单独目录下:
        data_dir/<code>/daily.csv

        Args:
            data_dict: {code: DataFrame} 股票数据
            output_dir: 输出目录

        Returns:
            输出目录路径
        """
        output_dir = Path(output_dir) if output_dir else self.data_dir / "qlib_csv"
        output_dir.mkdir(parents=True, exist_ok=True)

        for code, df in data_dict.items():
            inst_dir = output_dir / code
            inst_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(str(inst_dir / "daily.csv"), index=False)

        logger.info("Qlib CSV 格式转换完成: %d 只股票 -> %s", len(data_dict), output_dir)
        return str(output_dir)

    def create_qlib_dataset(
        self,
        data_path: Optional[str] = None,
    ):
        """
        创建 Qlib DatasetH 数据集

        生成 Qlib 可加载的数据集处理器，支持 train/valid/test 分割。

        Args:
            data_path: bin格式数据目录路径

        Returns:
            tuple: (train_dataset, valid_dataset, test_dataset) 或 None (qlib未安装)
        """
        if not self._qlib_available:
            logger.error("qlib 未安装，无法创建 DatasetH。请安装: pip install qlib")
            return None

        try:
            from qlib.data.dataset import DatasetH
            from qlib.data.dataset.handler import DataHandlerLP

            data_path = Path(data_path) if data_path else self.data_dir / "qlib_bin"

            # 配置
            handler_conf = {
                "start_time": _split_conf.get("train_start", "2013-01-01"),
                "end_time": _split_conf.get("test_end", "2023-12-31"),
                "fit_start_time": _split_conf.get("train_start", "2013-01-01"),
                "fit_end_time": _split_conf.get("train_end", "2021-06-30"),
                "instruments": "csi300",
            }

            segments = {
                "train": (_split_conf.get("train_start", "2013-01-01"), _split_conf.get("train_end", "2021-06-30")),
                "valid": (_split_conf.get("valid_start", "2021-07-01"), _split_conf.get("valid_end", "2022-12-31")),
                "test": (_split_conf.get("test_start", "2023-01-01"), _split_conf.get("test_end", "2023-12-31")),
            }

            handler = DataHandlerLP(
                instruments="csi300",
                start_time=handler_conf["start_time"],
                end_time=handler_conf["end_time"],
                data_loader_kwargs={"data_dir": str(data_path)},
            )

            datasets = {}
            for name, (start, end) in segments.items():
                datasets[name] = DatasetH(
                    handler=handler,
                    segments={name: [start, end]},
                )

            logger.info("Qlib DatasetH 创建完成: train/valid/test")
            return datasets.get("train"), datasets.get("valid"), datasets.get("test")

        except Exception as e:
            logger.error("创建Qlib DatasetH失败: %s", str(e)[:200])
            return None

    def convert_to_qlib(
        self,
        data_dict: Dict[str, pd.DataFrame],
        features_df: Optional[pd.DataFrame] = None,
        labels_df: Optional[pd.DataFrame] = None,
        output_dir: Optional[str] = None,
        format: str = "bin",
    ) -> str:
        """
        一站式转换为Qlib格式

        包括面板构建 + Qlib格式输出。

        Args:
            data_dict: {code: DataFrame} 股票数据
            features_df: 特征DataFrame
            labels_df: 标签DataFrame
            output_dir: 输出目录
            format: "bin" 或 "csv"

        Returns:
            输出目录路径
        """
        # 构建面板
        panel = self.build_panel_data(data_dict, features_df, labels_df)

        # 将面板拆分回字典格式（含特征和标签）
        converted_dict = {}
        for code, grp in panel.groupby("code"):
            converted_dict[code] = grp.reset_index(drop=True)

        if format == "bin":
            return self.convert_to_qlib_bin(converted_dict, output_dir)
        elif format == "csv":
            return self.convert_to_qlib_csv(converted_dict, output_dir)
        else:
            raise ValueError(f"不支持的格式: {format}，可选: bin, csv")


# ---- 模块级便捷函数 ----

def convert_to_qlib(
    data_dict: Dict[str, pd.DataFrame],
    features_df: Optional[pd.DataFrame] = None,
    labels_df: Optional[pd.DataFrame] = None,
    output_dir: Optional[str] = None,
    format: str = "bin",
) -> Optional[str]:
    """便捷转换函数"""
    converter = QlibConverter()
    return converter.convert_to_qlib(data_dict, features_df, labels_df, output_dir, format)


def create_qlib_dataset(data_path: Optional[str] = None):
    """便捷创建Qlib数据集"""
    converter = QlibConverter()
    return converter.create_qlib_dataset(data_path)