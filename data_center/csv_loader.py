# -*- coding: utf-8 -*-
"""
CSV 数据加载器 - CsvDataLoader

支持从 CSV 文件目录加载 A 股日线数据，兼容 sh.XXXXXX.csv / sz.XXXXXX.csv 命名格式。
自动将 CSV 数据转换为系统标准格式（添加 code 列、标准化列名），
提供与 DuckDBStore 一致的接口，供 train.py / daily_run.py 直接使用。
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)


class CsvDataLoader:
    """
    CSV 数据加载器

    从指定目录加载所有 CSV 文件，每只股票一个文件。
    支持多种命名格式：
      - sh.600000.csv / sz.000001.csv（A股常见格式）
      - 600000.csv / 000001.csv（纯代码格式）
      - any_prefix_600000.csv

    自动识别股票代码（文件名中6位数字部分），统一列名，
    添加 code 列，过滤无效数据。

    Attributes:
        data_dir: CSV 文件目录
        start_date: 数据起始日期（含）
        end_date: 数据结束日期（含）
    """

    # 标准化列名映射（兼容常见中英文名称）
    COLUMN_MAP = {
        "date": "date",
        "日期": "date",
        "trade_date": "date",
        "open": "open",
        "开盘": "open",
        "Open": "open",
        "high": "high",
        "最高": "high",
        "High": "high",
        "low": "low",
        "最低": "low",
        "Low": "low",
        "close": "close",
        "收盘": "close",
        "Close": "close",
        "volume": "volume",
        "成交量": "volume",
        "Volume": "volume",
        "vol": "volume",
        "amount": "amount",
        "成交额": "amount",
        "Amount": "amount",
        "turnover": "turnover_rate",
        "换手率": "turnover_rate",
        "turnover_rate": "turnover_rate",
        "pct_chg": "pct_chg",
        "涨跌幅": "pct_chg",
        "pctChange": "pct_chg",
        "change": "pct_chg",
        "preclose": "preclose",
        "前收盘": "preclose",
        "PreClose": "preclose",
        "amplitude": "amplitude",
        "振幅": "amplitude",
    }

    # 必需列
    REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}

    def __init__(
        self,
        data_dir: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        """
        初始化 CSV 数据加载器

        Args:
            data_dir: CSV 文件目录路径，默认从 settings.yaml 读取
            start_date: 数据起始日期 (YYYY-MM-DD)，默认从 settings.yaml 读取
            end_date: 数据结束日期 (YYYY-MM-DD)，默认从 settings.yaml 读取
        """
        if data_dir is None:
            config_path = _PROJECT_ROOT / "config" / "settings.yaml"
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            data_dir = cfg.get("data_source", {}).get(
                "csv_dir", cfg.get("paths", {}).get("csv_dir", "D:/data")
            )

        self.data_dir = Path(data_dir)

        # 加载配置中的默认日期范围
        config_path = _PROJECT_ROOT / "config" / "settings.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            default_start = cfg.get("data_source", {}).get("start_date", "2005-01-01")
        else:
            default_start = "2005-01-01"

        self.start_date = start_date or default_start
        self.end_date = end_date or "2099-12-31"

        self._file_cache = None

        logger.info(
            "CsvDataLoader 初始化: dir=%s, range=%s ~ %s",
            self.data_dir, self.start_date, self.end_date
        )

    def _discover_files(self) -> List[Path]:
        """发现所有 CSV 文件，提取股票代码。"""
        if self._file_cache is not None:
            return self._file_cache

        if not self.data_dir.exists():
            logger.warning("CSV 数据目录不存在: %s", self.data_dir)
            self._file_cache = []
            return []

        csv_files = sorted(self.data_dir.glob("*.csv"))
        result = []
        for f in csv_files:
            code = self._extract_code(f.name)
            if code:
                result.append((code, f))

        self._file_cache = result
        logger.info("发现 %d 个 CSV 文件, 目录: %s", len(result), self.data_dir)
        return result

    @staticmethod
    def _extract_code(filename: str) -> Optional[str]:
        """
        从文件名提取 6 位股票代码。

        支持格式：
          sh.600000.csv -> 600000
          sz.000001.csv -> 000001
          600000.csv    -> 600000
          SH600000.csv  -> 600000

        Returns:
            6 位纯数字代码字符串，或 None
        """
        name = Path(filename).stem  # 去掉 .csv
        # 去掉常见前缀分隔符
        for sep in [".", "_", "-"]:
            if sep in name:
                name = name.split(sep)[-1]

        # 提取 6 位数字
        import re
        match = re.search(r"(\d{6})", name)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名。"""
        rename_map = {}
        for col in df.columns:
            col_stripped = col.strip()
            if col_stripped in CsvDataLoader.COLUMN_MAP:
                rename_map[col] = CsvDataLoader.COLUMN_MAP[col_stripped]
        if rename_map:
            df = df.rename(columns=rename_map)
        return df

    def load_single(self, code: str) -> Optional[pd.DataFrame]:
        """
        加载单只股票数据。

        Args:
            code: 股票代码 (如 '600000')

        Returns:
            标准化后的 DataFrame（含 code 列），或 None
        """
        files = self._discover_files()
        for c, path in files:
            if c == code:
                return self._load_file(code, path)
        return None

    def _load_file(self, code: str, path: Path) -> Optional[pd.DataFrame]:
        """加载并标准化单个 CSV 文件。"""
        try:
            df = pd.read_csv(str(path))

            # 标准化列名
            df = self._normalize_columns(df)

            # 检查必需列
            missing = self.REQUIRED_COLUMNS - set(df.columns)
            if missing:
                logger.debug("跳过 %s: 缺少必需列 %s", code, missing)
                return None

            # 确保 date 列为字符串格式
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

            # 日期过滤
            df = df[
                (df["date"] >= self.start_date) & (df["date"] <= self.end_date)
            ].copy()

            # 去重
            df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

            if len(df) == 0:
                return None

            # 添加 code 列
            df["code"] = code

            # 确保数值列为 float
            for col in ["open", "high", "low", "close", "volume", "amount"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # 添加派生列
            if "amount" not in df.columns:
                df["amount"] = df["volume"] * df["close"]

            if "pct_chg" not in df.columns and len(df) >= 2:
                df["pct_chg"] = df["close"].pct_change() * 100

            if "turnover_rate" not in df.columns:
                df["turnover_rate"] = np.nan

            if "amplitude" not in df.columns:
                df["amplitude"] = np.nan

            if "preclose" not in df.columns and len(df) >= 2:
                df["preclose"] = df["close"].shift(1)

            # is_st 和 is_suspended
            df["is_st"] = False
            df["is_suspended"] = (df["volume"] == 0) | df["volume"].isna()

            return df

        except Exception as e:
            logger.debug("加载 %s 失败: %s", code, str(e)[:100])
            return None

    def get_stock_list(self) -> List[str]:
        """
        获取所有股票代码列表。

        Returns:
            股票代码字符串列表
        """
        files = self._discover_files()
        return [code for code, _ in files]

    def load_all(self, sample_size: int = 0, random_seed: int = 42) -> Dict[str, pd.DataFrame]:
        """
        加载所有（或采样）股票数据。

        Args:
            sample_size: 采样数量，0 或负数表示加载全部
            random_seed: 随机种子

        Returns:
            Dict[str, DataFrame]，key 为股票代码
        """
        files = self._discover_files()

        if sample_size > 0 and len(files) > sample_size:
            import random
            random.seed(random_seed)
            files = random.sample(files, sample_size)
            logger.info("采样 %d 只股票 (共 %d)", sample_size, len(files))

        data = {}
        loaded = 0
        failed = 0

        for code, path in files:
            df = self._load_file(code, path)
            if df is not None and not df.empty:
                data[code] = df
                loaded += 1
            else:
                failed += 1

        logger.info("CSV 数据加载完成: 成功 %d, 失败 %d", loaded, failed)
        return data

    def load_daily_data(self, code: str, start: Optional[str] = None, end: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        加载单只股票日线数据（兼容 DuckDBStore 接口）。

        Args:
            code: 股票代码
            start: 起始日期
            end: 结束日期

        Returns:
            DataFrame 或 None
        """
        # 临时覆盖日期范围
        orig_start, orig_end = self.start_date, self.end_date
        if start:
            self.start_date = start
        if end:
            self.end_date = end

        try:
            return self.load_single(code)
        finally:
            self.start_date, self.end_date = orig_start, orig_end

    def get_stock_price(self, code: str, start: Optional[str] = None, end: Optional[str] = None) -> Optional[pd.DataFrame]:
        """兼容 DuckDBStore 接口的别名。"""
        return self.load_daily_data(code, start, end)

    def get_price_history_multi(
        self,
        codes: List[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        批量加载多只股票数据（兼容 DuckDBStore 接口）。

        Returns:
            DataFrame with code 列
        """
        orig_start, orig_end = self.start_date, self.end_date
        if start:
            self.start_date = start
        if end:
            self.end_date = end

        try:
            dfs = []
            for code in codes:
                df = self.load_single(code)
                if df is not None:
                    if fields:
                        available = [c for c in fields if c in df.columns]
                        df = df[["code", "date"] + available]
                    dfs.append(df)

            if dfs:
                result = pd.concat(dfs, ignore_index=True)
                return result
            return pd.DataFrame()
        finally:
            self.start_date, self.end_date = orig_start, orig_end
