"""
时间序列数据集分割器

严格按时间顺序分割 train/valid/test，不进行随机打乱。
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict

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

_split_conf = _config.get("dataset_split", {})


class TimeSplitter:
    """
    时间序列数据集分割器

    严格按照时间顺序划分训练集、验证集和测试集。
    不进行随机打乱，确保不引入前视偏差。

    Attributes:
        train_start: 训练集开始日期
        train_end: 训练集结束日期
        valid_start: 验证集开始日期
        valid_end: 验证集结束日期
        test_start: 测试集开始日期
        test_end: 测试集结束日期
        backtest_start: 回测开始日期
        backtest_end: 回测结束日期
    """

    def __init__(
        self,
        train_start: Optional[str] = None,
        train_end: Optional[str] = None,
        valid_start: Optional[str] = None,
        valid_end: Optional[str] = None,
        test_start: Optional[str] = None,
        test_end: Optional[str] = None,
        backtest_start: Optional[str] = None,
        backtest_end: Optional[str] = None,
    ):
        self.train_start = train_start or _split_conf.get("train_start", "2013-01-01")
        self.train_end = train_end or _split_conf.get("train_end", "2021-06-30")
        self.valid_start = valid_start or _split_conf.get("valid_start", "2021-07-01")
        self.valid_end = valid_end or _split_conf.get("valid_end", "2022-12-31")
        self.test_start = test_start or _split_conf.get("test_start", "2023-01-01")
        self.test_end = test_end or _split_conf.get("test_end", "2023-12-31")
        self.backtest_start = backtest_start or _split_conf.get("backtest_start", "2024-01-01")
        self.backtest_end = backtest_end or _split_conf.get("backtest_end", "2025-06-30")

    def _get_dates_from_df(self, df: pd.DataFrame) -> pd.DatetimeIndex:
        """从DataFrame中提取日期列"""
        if "date" in df.columns:
            date_series = pd.to_datetime(df["date"])
        elif isinstance(df.index, pd.DatetimeIndex):
            date_series = df.index.to_series()
        else:
            raise ValueError("DataFrame缺少'date'列或DatetimeIndex索引")
        return pd.DatetimeIndex(date_series)

    def _get_dates_from_array(self, dates) -> pd.DatetimeIndex:
        """从多种格式中获取日期"""
        if isinstance(dates, pd.DatetimeIndex):
            return dates
        if isinstance(dates, pd.Series):
            return pd.DatetimeIndex(pd.to_datetime(dates))
        if isinstance(dates, list):
            return pd.DatetimeIndex(pd.to_datetime(dates))
        raise ValueError(f"不支持的日期格式: {type(dates)}")

    def split_train_test(
        self,
        dates,
        config: Optional[Dict] = None,
    ) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
        """
        将日期分为训练集和测试集（最简单分割）

        Args:
            dates: 日期序列 (DatetimeIndex, Series, 或 list-like)
            config: 可选覆盖配置，如 {"train_end": "2021-06-30", "test_start": "2021-07-01"}

        Returns:
            (train_dates, test_dates)
        """
        dates = self._get_dates_from_array(dates)

        c = config or {}
        train_end = c.get("train_end", self.train_end)
        test_start = c.get("test_start", self.test_start)
        test_end = c.get("test_end", None)

        train_dates = dates[(dates <= train_end)]
        test_mask = dates >= test_start
        if test_end:
            test_mask = test_mask & (dates <= test_end)
        test_dates = dates[test_mask]

        self._print_summary("Train/Test", train_dates, test_dates)
        return train_dates, test_dates

    def split_train_valid_test(
        self,
        dates,
        config: Optional[Dict] = None,
    ) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
        """
        将日期分为训练集、验证集和测试集

        Args:
            dates: 日期序列
            config: 可选覆盖配置

        Returns:
            (train_dates, valid_dates, test_dates)
        """
        dates = self._get_dates_from_array(dates)

        c = config or {}
        train_start = c.get("train_start", self.train_start)
        train_end = c.get("train_end", self.train_end)
        valid_start = c.get("valid_start", self.valid_start)
        valid_end = c.get("valid_end", self.valid_end)
        test_start = c.get("test_start", self.test_start)
        test_end = c.get("test_end", self.test_end)

        train_dates = dates[(dates >= train_start) & (dates <= train_end)]
        valid_dates = dates[(dates >= valid_start) & (dates <= valid_end)]
        test_dates = dates[(dates >= test_start) & (dates <= test_end)]

        self._print_summary("Train/Valid/Test", train_dates, valid_dates, test_dates)
        return train_dates, valid_dates, test_dates

    def split_dataframe(
        self,
        df: pd.DataFrame,
        mode: str = "train_valid_test",
        config: Optional[Dict] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        分割 DataFrame

        Args:
            df: 包含date列的DataFrame
            mode: "train_test" 或 "train_valid_test"
            config: 可选覆盖配置

        Returns:
            {"train": df_train, "valid": df_valid, "test": df_test}
            或 {"train": df_train, "test": df_test}
        """
        dates = self._get_dates_from_df(df)

        if mode == "train_test":
            train_dates, test_dates = self.split_train_test(dates, config)
            train_df = df[df["date"].isin(train_dates)].copy() if "date" in df.columns else df.loc[df.index.isin(train_dates)].copy()
            test_df = df[df["date"].isin(test_dates)].copy() if "date" in df.columns else df.loc[df.index.isin(test_dates)].copy()
            return {"train": train_df, "test": test_df}

        elif mode == "train_valid_test":
            train_dates, valid_dates, test_dates = self.split_train_valid_test(dates, config)
            train_df = df[df["date"].isin(train_dates)].copy() if "date" in df.columns else df.loc[df.index.isin(train_dates)].copy()
            valid_df = df[df["date"].isin(valid_dates)].copy() if "date" in df.columns else df.loc[df.index.isin(valid_dates)].copy()
            test_df = df[df["date"].isin(test_dates)].copy() if "date" in df.columns else df.loc[df.index.isin(test_dates)].copy()
            return {"train": train_df, "valid": valid_df, "test": test_df}

        else:
            raise ValueError(f"不支持的分割模式: {mode}，可选: train_test, train_valid_test")

    def split_dataframe_by_stock(
        self,
        df: pd.DataFrame,
        mode: str = "train_valid_test",
        config: Optional[Dict] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        按股票分组后分割DataFrame

        适用于包含多只股票的面板数据。

        Args:
            df: 包含 code, date 列的DataFrame
            mode: 分割模式
            config: 可选覆盖配置

        Returns:
            {"train": df_train, "valid": df_valid, "test": df_test}
        """
        if "code" not in df.columns:
            logger.warning("DataFrame无'code'列，使用整体分割")
            return self.split_dataframe(df, mode, config)

        if mode == "train_valid_test":
            train_start = config.get("train_start", self.train_start) if config else self.train_start
            train_end = config.get("train_end", self.train_end) if config else self.train_end
            valid_start = config.get("valid_start", self.valid_start) if config else self.valid_start
            valid_end = config.get("valid_end", self.valid_end) if config else self.valid_end
            test_start = config.get("test_start", self.test_start) if config else self.test_start
            test_end = config.get("test_end", self.test_end) if config else self.test_end

            train_df = df[(df["date"] >= train_start) & (df["date"] <= train_end)].copy()
            valid_df = df[(df["date"] >= valid_start) & (df["date"] <= valid_end)].copy()
            test_df = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()

            self._print_split_summary(train_df, valid_df, test_df)
            return {"train": train_df, "valid": valid_df, "test": test_df}

        elif mode == "train_test":
            train_end = config.get("train_end", self.train_end) if config else self.train_end
            test_start = config.get("test_start", self.test_start) if config else self.test_start

            train_df = df[df["date"] <= train_end].copy()
            test_df = df[df["date"] >= test_start].copy()

            self._print_split_summary(train_df, test_df)
            return {"train": train_df, "test": test_df}

        else:
            raise ValueError(f"不支持的分割模式: {mode}")

    def _print_summary(self, mode: str, *split_dates):
        """打印日期分割摘要"""
        names = []
        if "Train/Test" in mode:
            names = ["Train", "Test"]
        elif "Train/Valid/Test" in mode:
            names = ["Train", "Valid", "Test"]
        else:
            names = [f"Split{i}" for i in range(len(split_dates))]

        logger.info("=" * 50)
        logger.info("时间分割摘要 [%s]", mode)
        logger.info("-" * 50)
        for name, dates in zip(names, split_dates):
            if len(dates) > 0:
                logger.info(
                    "  %s: %d 个交易日, %s ~ %s",
                    name,
                    len(dates),
                    dates.min().strftime("%Y-%m-%d"),
                    dates.max().strftime("%Y-%m-%d"),
                )
            else:
                logger.info("  %s: 0 个交易日 (空)", name)
        logger.info("=" * 50)

    def _print_split_summary(self, *split_dfs):
        """打印DataFrame分割摘要"""
        names = []
        if len(split_dfs) == 3:
            names = ["Train", "Valid", "Test"]
        elif len(split_dfs) == 2:
            names = ["Train", "Test"]
        else:
            names = [f"Split{i}" for i in range(len(split_dfs))]

        logger.info("=" * 50)
        logger.info("时间分割摘要 (DataFrame)")
        logger.info("-" * 50)

        total_rows = sum(len(df) for df in split_dfs)
        for name, df in zip(names, split_dfs):
            n_stocks = df["code"].nunique() if "code" in df.columns else 1
            date_min = df["date"].min() if "date" in df.columns else "N/A"
            date_max = df["date"].max() if "date" in df.columns else "N/A"
            ratio = 100 * len(df) / total_rows if total_rows > 0 else 0
            logger.info(
                "  %s: %d 行, %d 只股票, %s ~ %s (%.1f%%)",
                name, len(df), n_stocks, str(date_min)[:10], str(date_max)[:10], ratio,
            )
        logger.info("=" * 50)


def split_train_test(dates, config: Optional[Dict] = None):
    """便捷函数: 分割为 train/test"""
    splitter = TimeSplitter()
    return splitter.split_train_test(dates, config)


def split_train_valid_test(dates, config: Optional[Dict] = None):
    """便捷函数: 分割为 train/valid/test"""
    splitter = TimeSplitter()
    return splitter.split_train_valid_test(dates, config)