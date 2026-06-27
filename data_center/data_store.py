# -*- coding: utf-8 -*-
"""
统一数据存储接口 (Unified DataStore)

屏蔽底层数据格式差异（Parquet / CSV / Qlib），向上层提供一致的数据访问接口。
上层代码（train.py / daily_run.py / generate_charts.py / backtest）统一通过 DataStore 访问数据，
修改 settings.yaml 中的 data_source.data_format 即可切换底层格式，无需改动业务代码。

支持的格式：
  - "parquet": 从 parquet_dir 读取 .parquet 文件（DuckDB 后端）
  - "csv":     从 csv_dir 读取 .csv 文件（CsvDataLoader 后端）
  - "qlib":    从 qlib_dir 读取 Qlib bin 格式（Qlib 后端）

使用方式：
  >>> from data_center.data_store import DataStore
  >>> store = DataStore()  # 自动根据 settings.yaml 创建对应后端
  >>> codes = store.get_stock_list()
  >>> df = store.get_stock_price("600000", "2023-01-01", "2024-01-01")
  >>> data_dict = store.load_all(sample_size=100)
"""

import os
import sys
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# 项目根目录
_PROJECT_ROOT = Path(__file__).parent.parent


def _load_settings() -> dict:
    """加载 settings.yaml 配置。"""
    config_path = _PROJECT_ROOT / "config" / "settings.yaml"
    if not config_path.exists():
        logger.warning("settings.yaml 不存在: %s", config_path)
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class BaseDataStore(ABC):
    """
    数据存储抽象基类。

    所有底层数据格式（Parquet/CSV/Qlib）都必须实现此接口，
    确保上层代码可以以统一方式访问任意格式的数据。
    """

    def __init__(self, start_date: Optional[str] = None, end_date: Optional[str] = None):
        self.start_date = start_date
        self.end_date = end_date

    @abstractmethod
    def get_stock_list(self) -> List[str]:
        """返回所有股票代码列表。"""
        pass

    @abstractmethod
    def get_stock_price(self, code: str, start: Optional[str] = None, end: Optional[str] = None) -> Optional[pd.DataFrame]:
        """加载单只股票的日线数据。"""
        pass

    @abstractmethod
    def load_all(self, sample_size: int = 0, random_seed: int = 42) -> Dict[str, pd.DataFrame]:
        """批量加载所有（或采样）股票数据，返回 Dict[code, DataFrame]。"""
        pass

    @abstractmethod
    def get_price_history_multi(self, codes: List[str], start: Optional[str] = None, end: Optional[str] = None,
                                 fields: Optional[List[str]] = None) -> pd.DataFrame:
        """批量加载多只股票数据，返回合并后的 DataFrame（含 code 列）。"""
        pass

    def load_single(self, code: str) -> Optional[pd.DataFrame]:
        """加载单只股票（兼容 csv_loader 接口）。"""
        return self.get_stock_price(code, self.start_date, self.end_date)


class ParquetDataStore(BaseDataStore):
    """Parquet + DuckDB 后端。"""

    def __init__(self, parquet_dir: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
        super().__init__(start_date, end_date)
        self.parquet_dir = Path(parquet_dir)
        self._store = None

    def _get_store(self):
        if self._store is None:
            from data_center.duckdb_store import DuckDBStore
            self._store = DuckDBStore(str(self.parquet_dir))
        return self._store

    def get_stock_list(self) -> List[str]:
        # 从 parquet 文件名提取代码
        files = list(self.parquet_dir.glob("*.parquet"))
        return sorted([f.stem for f in files])

    def get_stock_price(self, code: str, start: Optional[str] = None, end: Optional[str] = None) -> Optional[pd.DataFrame]:
        store = self._get_store()
        return store.get_stock_price(code, start or self.start_date, end or self.end_date)

    def load_all(self, sample_size: int = 0, random_seed: int = 42) -> Dict[str, pd.DataFrame]:
        codes = self.get_stock_list()
        if sample_size > 0 and len(codes) > sample_size:
            import random
            random.seed(random_seed)
            codes = random.sample(codes, sample_size)
        data = {}
        for code in codes:
            df = self.get_stock_price(code)
            if df is not None and not df.empty:
                data[code] = df
        return data

    def get_price_history_multi(self, codes: List[str], start: Optional[str] = None, end: Optional[str] = None,
                                 fields: Optional[List[str]] = None) -> pd.DataFrame:
        store = self._get_store()
        return store.get_price_history_multi(codes, start or self.start_date, end or self.end_date, fields)


class CsvDataStore(BaseDataStore):
    """CSV 后端，复用现有的 CsvDataLoader。"""

    def __init__(self, csv_dir: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
        super().__init__(start_date, end_date)
        from data_center.csv_loader import CsvDataLoader
        self._loader = CsvDataLoader(data_dir=csv_dir, start_date=start_date, end_date=end_date)

    def get_stock_list(self) -> List[str]:
        return self._loader.get_stock_list()

    def get_stock_price(self, code: str, start: Optional[str] = None, end: Optional[str] = None) -> Optional[pd.DataFrame]:
        return self._loader.load_daily_data(code, start, end)

    def load_all(self, sample_size: int = 0, random_seed: int = 42) -> Dict[str, pd.DataFrame]:
        return self._loader.load_all(sample_size=sample_size, random_seed=random_seed)

    def get_price_history_multi(self, codes: List[str], start: Optional[str] = None, end: Optional[str] = None,
                                 fields: Optional[List[str]] = None) -> pd.DataFrame:
        return self._loader.get_price_history_multi(codes, start, end, fields)


class QlibDataStore(BaseDataStore):
    """Qlib bin 格式后端。"""

    def __init__(self, qlib_dir: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
        super().__init__(start_date, end_date)
        self.qlib_dir = qlib_dir
        try:
            from qlib import init
            from qlib.data import D
            init(provider_uri=qlib_dir, region="cn")
            self._D = D
        except Exception as e:
            logger.error("Qlib 初始化失败: %s", e)
            self._D = None

    def get_stock_list(self) -> List[str]:
        # 从 instruments 文件读取
        inst_file = Path(self.qlib_dir) / "instruments" / "all.txt"
        if not inst_file.exists():
            return []
        with open(inst_file) as f:
            return [line.strip().split("\t")[0] for line in f if line.strip()]

    def get_stock_price(self, code: str, start: Optional[str] = None, end: Optional[str] = None) -> Optional[pd.DataFrame]:
        if self._D is None:
            return None
        try:
            df = self._D.features([code], fields=["$close", "$open", "$high", "$low", "$volume"],
                                  start_time=start or self.start_date, end_time=end or self.end_date)
            if df is None or df.empty:
                return None
            df = df.reset_index()
            df.rename(columns={"instrument": "code", "datetime": "date"}, inplace=True)
            return df
        except Exception as e:
            logger.debug("Qlib 加载 %s 失败: %s", code, e)
            return None

    def load_all(self, sample_size: int = 0, random_seed: int = 42) -> Dict[str, pd.DataFrame]:
        codes = self.get_stock_list()
        if sample_size > 0 and len(codes) > sample_size:
            import random
            random.seed(random_seed)
            codes = random.sample(codes, sample_size)
        data = {}
        for code in codes:
            df = self.get_stock_price(code)
            if df is not None and not df.empty:
                data[code] = df
        return data

    def get_price_history_multi(self, codes: List[str], start: Optional[str] = None, end: Optional[str] = None,
                                 fields: Optional[List[str]] = None) -> pd.DataFrame:
        if self._D is None:
            return pd.DataFrame()
        try:
            qlib_fields = fields or ["$close", "$open", "$high", "$low", "$volume"]
            df = self._D.features(codes, fields=qlib_fields,
                                  start_time=start or self.start_date, end_time=end or self.end_date)
            df = df.reset_index()
            df.rename(columns={"instrument": "code", "datetime": "date"}, inplace=True)
            return df
        except Exception as e:
            logger.error("Qlib 批量加载失败: %s", e)
            return pd.DataFrame()


class DataStore:
    """
    统一数据存储工厂类。

    根据 settings.yaml 中 data_source.data_format 的值，
    自动创建对应的数据后端实例。所有上层代码统一使用此类。

    Args:
        start_date: 数据起始日期
        end_date: 数据结束日期
        config: 可选，传入已加载的配置字典
    """

    def __new__(cls, start_date: Optional[str] = None, end_date: Optional[str] = None,
                config: Optional[dict] = None):
        cfg = config or _load_settings()
        ds_cfg = cfg.get("data_source", {})
        fmt = ds_cfg.get("data_format", ds_cfg.get("primary", "csv"))

        logger.info("DataStore 初始化: format=%s", fmt)

        if fmt == "parquet":
            parquet_dir = ds_cfg.get("parquet_dir", str(_PROJECT_ROOT / "stock-ai" / "data" / "daily"))
            return ParquetDataStore(parquet_dir, start_date, end_date)

        elif fmt == "csv":
            csv_dir = ds_cfg.get("csv_dir", "D:/data")
            return CsvDataStore(csv_dir, start_date, end_date)

        elif fmt == "qlib":
            qlib_dir = ds_cfg.get("qlib_dir", str(_PROJECT_ROOT / "qlib_data" / "cn_data"))
            return QlibDataStore(qlib_dir, start_date, end_date)

        else:
            logger.warning("未知的 data_format=%s，回退到 CSV", fmt)
            csv_dir = ds_cfg.get("csv_dir", "D:/data")
            return CsvDataStore(csv_dir, start_date, end_date)
