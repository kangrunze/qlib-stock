"""
因子存储与版本管理 - FactorStore

提供因子数据的持久化存储、版本管理和检索功能。
使用 Parquet 格式按日期分区存储，支持高效读写和宽矩阵构建。

存储结构:
    features_dir/
        v1.0/
            2020-01-02.parquet
            2020-01-03.parquet
            ...
            _manifest.json  (元数据)
        v1.1/
            ...
    versions.json  (版本列表)

输入/输出: Dict[str, pd.DataFrame] (股票代码 -> 因子 DataFrame)

Author: stock-ai quantitative trading project
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config.settings import SettingsManager

logger = logging.getLogger(__name__)


# ==============================================================================
# FactorStore 类
# ==============================================================================

class FactorStore:
    """因子存储与版本管理器。

    负责因子的持久化存储、加载、版本控制和宽矩阵构建。
    使用 Parquet 格式，按日期分区存储。

    Attributes:
        settings: 配置管理器实例
        features_dir: 因子存储根目录
        compression: Parquet 压缩算法
        default_version: 默认版本号

    Usage:
        >>> store = FactorStore()
        >>> store.save_factors(factors_dict, version="v1.0", date="2024-01-02")
        >>> loaded = store.load_factors(["000001", "000002"], "2024-01-01", "2024-01-31")
        >>> matrix = store.build_factor_matrix(factors_dict, "2024-01-01", "2024-01-31")
    """

    def __init__(self, features_dir: str = None, version: str = None):
        """初始化因子存储管理器。

        Args:
            features_dir: 因子存储根目录。如果为 None，使用配置中的 features_dir
            version: 默认版本号。如果为 None，使用配置中的 default_version
        """
        self.settings = SettingsManager()

        if features_dir is None:
            features_dir = self.settings.features_dir
        self.features_dir = features_dir

        if version is None:
            version = self.settings.default_version
        self.default_version = version

        self.compression = self.settings.store_compression

        # 确保存储目录存在
        os.makedirs(self.features_dir, exist_ok=True)

    # ===== 目录管理 =====

    def _get_version_dir(self, version: str = None) -> str:
        """获取指定版本的存储目录路径。

        Args:
            version: 版本号，如果为 None 使用默认版本

        Returns:
            版本目录路径
        """
        if version is None:
            version = self.default_version
        return os.path.join(self.features_dir, version)

    def _ensure_version_dir(self, version: str = None):
        """确保版本目录存在。

        Args:
            version: 版本号
        """
        version_dir = self._get_version_dir(version)
        os.makedirs(version_dir, exist_ok=True)

    # ===== 保存因子 =====

    def save_factors(
        self,
        factors_dict: Dict[str, pd.DataFrame],
        version: str = None,
        date: Union[str, datetime] = None,
    ) -> str:
        """保存单日因子数据为 Parquet 文件。

        将某一天所有股票的因子数据保存到一个 Parquet 文件中。
        文件中包含 'stock' 列标记股票代码。

        Args:
            factors_dict: {股票代码: 因子 DataFrame}。每个 DataFrame
                          应有相同的因子列和相同的索引（当日）。
            version: 版本号，如果为 None 使用默认版本
            date: 日期，如果为 None 使用 DataFrame 的第一个索引日期

        Returns:
            保存的文件路径

        Raises:
            ValueError: 如果 factors_dict 为空
        """
        if not factors_dict:
            raise ValueError("factors_dict 为空，无法保存")

        if version is None:
            version = self.default_version

        self._ensure_version_dir(version)

        # 收集所有股票的当日因子数据
        daily_data = []
        for stock, df in factors_dict.items():
            if df.empty:
                continue
            if date is not None:
                # 提取指定日期的数据
                if date in df.index:
                    row = df.loc[[date]].copy()
                else:
                    logger.warning("股票 %s 在日期 %s 无数据", stock, date)
                    continue
            else:
                # 使用 DataFrame 的第一行
                row = df.head(1).copy()

            row["stock"] = stock
            daily_data.append(row)

        if not daily_data:
            logger.warning("没有可保存的因子数据")
            return ""

        # 合并所有股票
        combined = pd.concat(daily_data, axis=0, ignore_index=False)
        combined = combined.reset_index()

        # 确定文件日期
        if date is not None:
            if isinstance(date, datetime):
                date_str = date.strftime("%Y-%m-%d")
            else:
                date_str = str(date)[:10]
        elif "date" in combined.columns:
            date_str = str(combined["date"].iloc[0])[:10]
        elif not combined.empty:
            # 从 index 中取第一个
            idx_date = combined.index[0] if not isinstance(combined.index, pd.RangeIndex) else combined["index"].iloc[0]
            date_str = str(idx_date)[:10]
        else:
            date_str = datetime.now().strftime("%Y-%m-%d")

        version_dir = self._get_version_dir(version)
        file_path = os.path.join(version_dir, f"{date_str}.parquet")

        # 写入 Parquet
        table = pa.Table.from_pandas(combined, preserve_index=True)
        pq.write_table(
            table,
            file_path,
            compression=self.compression,
        )

        logger.info(
            "保存因子: 版本=%s, 日期=%s, 股票数=%d, 路径=%s",
            version,
            date_str,
            len(daily_data),
            file_path,
        )

        # 更新 manifest
        self._update_manifest(version, date_str, len(daily_data))

        return file_path

    def save_factors_batch(
        self,
        factors_dict: Dict[str, pd.DataFrame],
        version: str = None,
        date_range: Tuple[str, str] = None,
    ) -> List[str]:
        """批量保存多日因子数据。

        将 factors_dict 按日期拆分，每个日期保存一个 Parquet 文件。

        Args:
            factors_dict: {股票代码: 因子 DataFrame}
            version: 版本号
            date_range: (start_date, end_date) 限制保存的日期范围

        Returns:
            保存的文件路径列表
        """
        if version is None:
            version = self.default_version

        # 收集所有日期
        all_dates = set()
        for df in factors_dict.values():
            if not df.empty:
                all_dates.update(df.index)

        all_dates = sorted(all_dates)

        # 过滤日期范围
        if date_range:
            start, end = date_range
            all_dates = [d for d in all_dates if start <= str(d)[:10] <= end]

        saved_files = []
        total_dates = len(all_dates)

        for idx, date in enumerate(all_dates):
            # 提取当日每只股票的因子数据
            daily_factors = {}
            for stock, df in factors_dict.items():
                if date in df.index:
                    daily_factors[stock] = df.loc[[date]]

            if daily_factors:
                file_path = self.save_factors(daily_factors, version=version, date=date)
                if file_path:
                    saved_files.append(file_path)

            if (idx + 1) % 500 == 0 or idx == total_dates - 1:
                logger.info(
                    "批量保存进度: %d/%d 天",
                    idx + 1,
                    total_dates,
                )

        logger.info(
            "批量保存完成: 共 %d 个文件, 版本=%s",
            len(saved_files),
            version,
        )
        return saved_files

    # ===== 加载因子 =====

    def load_factors(
        self,
        stock_codes: List[str],
        start_date: str,
        end_date: str,
        version: str = None,
    ) -> Dict[str, pd.DataFrame]:
        """加载指定股票和时间范围的因子数据。

        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            version: 版本号，如果为 None 使用默认版本

        Returns:
            {股票代码: 因子 DataFrame}
        """
        if version is None:
            version = self.default_version

        version_dir = self._get_version_dir(version)

        if not os.path.exists(version_dir):
            logger.warning("版本目录不存在: %s", version_dir)
            return {stock: pd.DataFrame() for stock in stock_codes}

        # 获取日期范围内的 Parquet 文件
        parquet_files = self._list_parquet_files_in_range(
            version_dir, start_date, end_date
        )

        if not parquet_files:
            logger.warning(
                "日期范围内无因子数据: %s ~ %s, 版本=%s",
                start_date,
                end_date,
                version,
            )
            return {stock: pd.DataFrame() for stock in stock_codes}

        # 初始化结果
        stock_data: Dict[str, list] = {stock: [] for stock in stock_codes}

        for file_path in parquet_files:
            try:
                df = pd.read_parquet(file_path)
            except Exception as e:
                logger.warning("读取文件失败 %s: %s", file_path, e)
                continue

            if "stock" not in df.columns:
                logger.warning("文件缺少 stock 列: %s", file_path)
                continue

            # 筛选目标股票
            filtered = df[df["stock"].isin(stock_codes)]

            for stock in stock_codes:
                stock_df = filtered[filtered["stock"] == stock].copy()
                if not stock_df.empty:
                    # 移除 stock 列并设置索引
                    stock_df = stock_df.drop(columns=["stock"], errors="ignore")
                    # 尝试从 date 或 index 列设置索引
                    if "date" in stock_df.columns:
                        stock_df = stock_df.set_index("date")
                    elif "index" in stock_df.columns:
                        stock_df = stock_df.set_index("index")
                    stock_data[stock].append(stock_df)

        # 合并各股票的数据
        result = {}
        for stock, dfs in stock_data.items():
            if dfs:
                combined = pd.concat(dfs, axis=0)
                combined = combined.sort_index()
                result[stock] = combined
            else:
                result[stock] = pd.DataFrame()

        loaded_stocks = sum(1 for v in result.values() if not v.empty)
        logger.info(
            "加载因子: 版本=%s, %d/%d 只股票有数据, %d 个文件",
            version,
            loaded_stocks,
            len(stock_codes),
            len(parquet_files),
        )

        return result

    def _list_parquet_files_in_range(
        self,
        version_dir: str,
        start_date: str,
        end_date: str,
    ) -> List[str]:
        """列出日期范围内的 Parquet 文件。

        Args:
            version_dir: 版本目录
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            文件路径列表
        """
        files = []
        for fname in os.listdir(version_dir):
            if fname.endswith(".parquet"):
                date_str = fname.replace(".parquet", "")
                if start_date <= date_str <= end_date:
                    files.append(os.path.join(version_dir, fname))
        return sorted(files)

    # ===== 版本管理 =====

    def get_latest_version(self) -> str:
        """获取最新的因子版本号。

        Returns:
            最新版本号，如果没有版本则返回默认版本
        """
        versions = self.list_versions()
        if versions:
            return versions[-1]
        return self.default_version

    def list_versions(self) -> List[str]:
        """列出所有可用的因子版本。

        Returns:
            版本号列表，按创建时间排序
        """
        versions = []
        if not os.path.exists(self.features_dir):
            return versions

        for item in os.listdir(self.features_dir):
            item_path = os.path.join(self.features_dir, item)
            if os.path.isdir(item_path) and os.path.exists(
                os.path.join(item_path, "_manifest.json")
            ):
                versions.append(item)

        return sorted(versions)

    def create_version(self, version: str) -> str:
        """创建新的因子版本。

        Args:
            version: 版本号

        Returns:
            创建的版本目录路径

        Raises:
            ValueError: 如果版本已存在
        """
        version_dir = self._get_version_dir(version)
        if os.path.exists(version_dir):
            raise ValueError(f"版本 {version} 已存在")

        os.makedirs(version_dir, exist_ok=True)
        self._save_version_metadata(version)
        logger.info("创建新版本: %s", version)
        return version_dir

    def delete_version(self, version: str) -> bool:
        """删除指定版本的所有因子数据。

        Args:
            version: 版本号

        Returns:
            是否删除成功

        Raises:
            ValueError: 如果版本不存在
        """
        version_dir = self._get_version_dir(version)
        if not os.path.exists(version_dir):
            raise ValueError(f"版本 {version} 不存在")

        import shutil
        shutil.rmtree(version_dir)
        logger.info("删除版本: %s", version)
        return True

    # ===== 元数据管理 =====

    def _update_manifest(self, version: str, date_str: str, stock_count: int):
        """更新版本 manifest 文件。

        记录已保存的日期和股票数量。

        Args:
            version: 版本号
            date_str: 日期字符串
            stock_count: 股票数量
        """
        version_dir = self._get_version_dir(version)
        manifest_path = os.path.join(version_dir, "_manifest.json")

        manifest = {}
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

        manifest.setdefault("dates", {})
        manifest["dates"][date_str] = {
            "stock_count": stock_count,
            "updated_at": datetime.now().isoformat(),
        }
        manifest["last_updated"] = datetime.now().isoformat()
        manifest["total_dates"] = len(manifest["dates"])
        manifest["version"] = version

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    def _save_version_metadata(self, version: str):
        """保存版本元数据。

        Args:
            version: 版本号
        """
        version_dir = self._get_version_dir(version)
        manifest_path = os.path.join(version_dir, "_manifest.json")
        manifest = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "dates": {},
            "total_dates": 0,
            "last_updated": datetime.now().isoformat(),
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    def get_version_info(self, version: str = None) -> dict:
        """获取版本信息。

        Args:
            version: 版本号，如果为 None 使用默认版本

        Returns:
            版本信息字典
        """
        if version is None:
            version = self.default_version

        version_dir = self._get_version_dir(version)
        manifest_path = os.path.join(version_dir, "_manifest.json")

        if not os.path.exists(manifest_path):
            return {"version": version, "exists": False}

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        return manifest

    # ===== 宽矩阵构建 =====

    def build_factor_matrix(
        self,
        factors_dict: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """构建宽因子矩阵 (stocks x dates x factors)，适合 ML 训练。

        输出格式: MultiIndex (date, stock) 的 DataFrame，
        列是因子名称。可直接用于 sklearn/LightGBM 等模型的 X 输入。

        Args:
            factors_dict: {股票代码: 因子 DataFrame}
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            宽格式因子矩阵 DataFrame，index 为 (date, stock) MultiIndex

        Raises:
            ValueError: 如果 factors_dict 为空
        """
        if not factors_dict:
            raise ValueError("factors_dict 为空")

        logger.info(
            "构建宽因子矩阵: %s ~ %s, 股票数=%d",
            start_date,
            end_date,
            len(factors_dict),
        )

        panels = []

        for stock, df in factors_dict.items():
            if df.empty:
                continue

            # 过滤日期范围
            df = df.sort_index()
            mask = (df.index >= start_date) & (df.index <= end_date)
            df = df.loc[mask]

            if df.empty:
                continue

            df = df.copy()
            df["stock"] = stock
            panels.append(df)

        if not panels:
            logger.warning("日期范围内无因子数据")
            return pd.DataFrame()

        # 合并所有股票
        combined = pd.concat(panels, axis=0)
        combined = combined.reset_index()

        # 重命名 index 列为 date
        index_col = combined.columns[0]
        if index_col != "date":
            combined = combined.rename(columns={index_col: "date"})

        # 设置 MultiIndex (date, stock)
        combined = combined.set_index(["date", "stock"])

        # 清理
        combined = combined.replace([np.inf, -np.inf], np.nan)

        logger.info(
            "宽矩阵构建完成: 形状=%s, 因子数=%d",
            combined.shape,
            len(combined.columns),
        )

        return combined

    def load_factor_matrix(
        self,
        stock_codes: List[str],
        start_date: str,
        end_date: str,
        version: str = None,
    ) -> pd.DataFrame:
        """从存储中加载并构建宽因子矩阵。

        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            version: 版本号

        Returns:
            宽格式因子矩阵 DataFrame
        """
        factors_dict = self.load_factors(
            stock_codes, start_date, end_date, version
        )
        return self.build_factor_matrix(factors_dict, start_date, end_date)

    # ===== 数据统计 =====

    def get_coverage_stats(
        self,
        stock_codes: List[str],
        start_date: str,
        end_date: str,
        version: str = None,
    ) -> dict:
        """获取因子数据覆盖率统计。

        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            version: 版本号

        Returns:
            统计信息字典
        """
        factors_dict = self.load_factors(
            stock_codes, start_date, end_date, version
        )

        stats = {
            "total_stocks": len(stock_codes),
            "stocks_with_data": 0,
            "stocks_without_data": 0,
            "total_records": 0,
            "avg_records_per_stock": 0,
            "factor_count": 0,
            "version": version or self.default_version,
        }

        total_records = 0
        stocks_with_data = 0
        factor_names = set()

        for stock in stock_codes:
            df = factors_dict.get(stock, pd.DataFrame())
            if not df.empty:
                stocks_with_data += 1
                total_records += len(df)
                factor_names.update(df.columns)

        stats["stocks_with_data"] = stocks_with_data
        stats["stocks_without_data"] = len(stock_codes) - stocks_with_data
        stats["total_records"] = total_records
        stats["factor_count"] = len(factor_names)
        stats["factor_names"] = sorted(factor_names)
        stats["coverage_ratio"] = (
            stocks_with_data / len(stock_codes) if stock_codes else 0
        )

        return stats

    def __repr__(self) -> str:
        return (
            f"FactorStore(dir='{self.features_dir}', "
            f"version='{self.default_version}', "
            f"compression='{self.compression}')"
        )


# ==============================================================================
# 便捷函数
# ==============================================================================

def save_factors(
    factors_dict: Dict[str, pd.DataFrame],
    version: str = None,
    date: Union[str, datetime] = None,
) -> str:
    """便捷函数：保存单日因子数据。

    Args:
        factors_dict: {股票代码: 因子 DataFrame}
        version: 版本号
        date: 日期

    Returns:
        保存的文件路径
    """
    store = FactorStore()
    return store.save_factors(factors_dict, version=version, date=date)


def load_factors(
    stock_codes: List[str],
    start_date: str,
    end_date: str,
    version: str = None,
) -> Dict[str, pd.DataFrame]:
    """便捷函数：加载因子数据。

    Args:
        stock_codes: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        version: 版本号

    Returns:
        {股票代码: 因子 DataFrame}
    """
    store = FactorStore()
    return store.load_factors(stock_codes, start_date, end_date, version=version)


def build_factor_matrix(
    factors_dict: Dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """便捷函数：构建宽因子矩阵。

    Args:
        factors_dict: {股票代码: 因子 DataFrame}
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        宽格式因子矩阵
    """
    store = FactorStore()
    return store.build_factor_matrix(factors_dict, start_date, end_date)