"""
数据清洗和预处理模块

提供缺失值处理、异常值裁剪、标准化、IC筛选和相关性筛选功能。
"""

import sys
import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
import yaml

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ---- 加载配置 ----
_CONFIG_PATH = _PROJECT_ROOT / "config" / "settings.yaml"
with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    _config = yaml.safe_load(f)

_features_conf = _config.get("features", {})
IC_THRESHOLD = _features_conf.get("ic_threshold", 0.02)
CORR_THRESHOLD = _features_conf.get("corr_threshold", 0.95)


class DataCleaner:
    """
    数据清洗和预处理

    提供以下功能:
    - 缺失值处理（组内前向填充 + 横截面中位数填充）
    - 异常值裁剪（Winsorize 1%/99%）
    - 截面标准化（Z-Score）
    - IC筛选（信息系数阈值过滤）
    - 相关性筛选（高相关性特征去重）

    Attributes:
        ic_threshold: IC绝对值阈值，低于此值的特征被过滤
        corr_threshold: 相关系数阈值，高于此值的特征对中保留一个
    """

    def __init__(
        self,
        ic_threshold: float = IC_THRESHOLD,
        corr_threshold: float = CORR_THRESHOLD,
    ):
        self.ic_threshold = ic_threshold
        self.corr_threshold = corr_threshold

    # ==================== 缺失值处理 ====================

    def handle_missing(
        self,
        df: pd.DataFrame,
        group_col: str = "code",
        date_col: str = "date",
    ) -> pd.DataFrame:
        """
        处理缺失值

        策略:
        1. 按股票内部前向填充（ffill），再后向填充（bfill）
        2. 剩余缺失值用横截面中位数填充

        Args:
            df: 输入DataFrame
            group_col: 分组列（股票代码）
            date_col: 日期列

        Returns:
            处理后的DataFrame
        """
        df = df.copy()

        # 排除非数值列
        feature_cols = [
            c for c in df.columns
            if c not in (group_col, date_col) and pd.api.types.is_numeric_dtype(df[c])
        ]

        if not feature_cols:
            logger.warning("未找到数值特征列")
            return df

        total_missing_before = df[feature_cols].isnull().sum().sum()

        # Step 1: 按股票分组内前向填充
        if group_col in df.columns:
            df = df.sort_values([group_col, date_col])
            for col in feature_cols:
                df[col] = df.groupby(group_col)[col].transform(
                    lambda x: x.ffill().bfill()
                )

        # Step 2: 剩余缺失值用横截面中位数填充
        for col in feature_cols:
            mask = df[col].isnull()
            if mask.any():
                median_val = df.groupby(date_col)[col].transform("median")
                df.loc[mask, col] = median_val.loc[mask]

        # 最终仍有缺失的用全局中位数
        for col in feature_cols:
            mask = df[col].isnull()
            if mask.any():
                df.loc[mask, col] = df[col].median()

        total_missing_after = df[feature_cols].isnull().sum().sum()
        logger.info(
            "缺失值处理: %d -> %d (减少 %.1f%%)",
            total_missing_before, total_missing_after,
            100 * (1 - total_missing_after / max(total_missing_before, 1))
        )

        return df

    def clean_features(self, df: pd.DataFrame, group_col: str = "code", date_col: str = "date") -> pd.DataFrame:
        """
        完整的数据清洗流程

        包含缺失值处理，但不自动执行标准化（需按需调用）。
        标准化的时机应在特征工程后、训练前。

        Args:
            df: 输入DataFrame
            group_col: 分组列
            date_col: 日期列

        Returns:
            清洗后的DataFrame
        """
        logger.info("开始数据清洗...")
        df = self.handle_missing(df, group_col=group_col, date_col=date_col)
        logger.info("数据清洗完成")
        return df

    # ==================== 异常值裁剪 ====================

    def winsorize(
        self,
        series: pd.Series,
        limits: Tuple[float, float] = (0.01, 0.01),
    ) -> pd.Series:
        """
        Winsorize 异常值裁剪

        将超出分位数边界的值裁剪到边界值。

        Args:
            series: 输入序列
            limits: (下界比例, 上界比例)，默认 (0.01, 0.01) 即1%/99%

        Returns:
            裁剪后的序列
        """
        if limits[0] <= 0 and limits[1] <= 0:
            return series

        result = series.copy()
        lower = series.quantile(limits[0]) if limits[0] > 0 else -np.inf
        upper = series.quantile(1 - limits[1]) if limits[1] > 0 else np.inf

        n_clipped_lower = (series < lower).sum()
        n_clipped_upper = (series > upper).sum()

        result = result.clip(lower=lower, upper=upper)

        if n_clipped_lower > 0 or n_clipped_upper > 0:
            logger.debug(
                "Winsorize: 下界裁剪=%d, 上界裁剪=%d (lower=%.4f, upper=%.4f)",
                n_clipped_lower, n_clipped_upper, lower, upper
            )

        return result

    def winsorize_dataframe(
        self,
        df: pd.DataFrame,
        limits: Tuple[float, float] = (0.01, 0.01),
        group_col: str = "date",
    ) -> pd.DataFrame:
        """
        对DataFrame按截面进行Winsorize

        Args:
            df: 输入DataFrame
            limits: 裁剪比例
            group_col: 截面分组列（默认按日期）

        Returns:
            裁剪后的DataFrame
        """
        df = df.copy()
        feature_cols = [
            c for c in df.columns
            if c != group_col and pd.api.types.is_numeric_dtype(df[c])
        ]

        for col in feature_cols:
            if group_col in df.columns:
                df[col] = df.groupby(group_col)[col].transform(
                    lambda x: self.winsorize(x, limits)
                )
            else:
                df[col] = self.winsorize(df[col], limits)

        return df

    # ==================== 标准化 ====================

    def standardize_cross_section(
        self,
        df: pd.DataFrame,
        group_col: str = "date",
        method: str = "zscore",
    ) -> pd.DataFrame:
        """
        横截面标准化（按日期分组）

        对每个交易日截面内的特征进行标准化。

        Args:
            df: 输入DataFrame
            group_col: 截面分组列（默认按日期）
            method: 标准化方法
                - "zscore": (x - mean) / std
                - "rank": 排名分位数标准化 (rank / N)
                - "minmax": (x - min) / (max - min)

        Returns:
            标准化后的DataFrame
        """
        df = df.copy()

        feature_cols = [
            c for c in df.columns
            if c != group_col and pd.api.types.is_numeric_dtype(df[c])
        ]

        if not feature_cols:
            return df

        if group_col not in df.columns:
            logger.warning("缺少分组列 %s，进行全局标准化", group_col)
            for col in feature_cols:
                if method == "zscore":
                    mean = df[col].mean()
                    std = df[col].std()
                    if std > 0:
                        df[col] = (df[col] - mean) / std
                    else:
                        df[col] = 0.0
                elif method == "rank":
                    df[col] = df[col].rank(pct=True)
                elif method == "minmax":
                    min_val = df[col].min()
                    max_val = df[col].max()
                    if max_val > min_val:
                        df[col] = (df[col] - min_val) / (max_val - min_val)
                    else:
                        df[col] = 0.0
            return df

        def _standardize_group(grp: pd.DataFrame) -> pd.DataFrame:
            grp = grp.copy()
            for col in feature_cols:
                if col not in grp.columns:
                    continue
                if method == "zscore":
                    mean = grp[col].mean()
                    std = grp[col].std()
                    if std and std > 0:
                        grp[col] = (grp[col] - mean) / std
                    else:
                        grp[col] = 0.0
                elif method == "rank":
                    grp[col] = grp[col].rank(pct=True)
                elif method == "minmax":
                    min_val = grp[col].min()
                    max_val = grp[col].max()
                    if max_val > min_val:
                        grp[col] = (grp[col] - min_val) / (max_val - min_val)
                    else:
                        grp[col] = 0.0
            return grp

        df = df.groupby(group_col, group_keys=False).apply(_standardize_group)
        logger.info("横截面标准化完成: method=%s, %d 个特征", method, len(feature_cols))
        return df

    # ==================== IC筛选 ====================

    def filter_by_ic(
        self,
        features: pd.DataFrame,
        labels: Union[pd.Series, pd.DataFrame],
        threshold: Optional[float] = None,
        date_col: str = "date",
    ) -> List[str]:
        """
        按 Information Coefficient (IC) 筛选特征

        计算每个特征与标签的Rank IC（截面Spearman相关系数），
        保留 |mean IC| >= threshold 的特征。

        Args:
            features: 特征DataFrame, 含date列
            labels: 标签Series或DataFrame
            threshold: IC绝对值阈值, None则使用默认配置
            date_col: 日期列

        Returns:
            保留的特征列名列表
        """
        threshold = threshold if threshold is not None else self.ic_threshold

        feature_cols = [
            c for c in features.columns
            if c != date_col and pd.api.types.is_numeric_dtype(features[c])
        ]

        if isinstance(labels, pd.DataFrame):
            # 找第一个label列
            label_cols = [c for c in labels.columns if c not in (date_col, "code")]
            if not label_cols:
                logger.warning("标签DataFrame中未找到标签列")
                return feature_cols
            label_col = label_cols[0]
            labels = labels[label_col]
        elif isinstance(labels, pd.Series):
            label_col = labels.name or "label"

        if date_col not in features.columns or len(feature_cols) == 0:
            return feature_cols

        ic_results = {}
        dates = sorted(features[date_col].unique())

        for feat in feature_cols:
            daily_ics = []
            for d in dates:
                mask = features[date_col] == d
                feat_vals = features.loc[mask, feat]
                label_vals = labels.loc[mask] if len(labels) == len(features) else labels.loc[mask]

                if feat_vals.notna().sum() < 10 or label_vals.notna().sum() < 10:
                    continue

                ic, _ = stats.spearmanr(
                    feat_vals.dropna().reindex(label_vals.dropna().index).dropna(),
                    label_vals.dropna().reindex(feat_vals.dropna().index).dropna()
                )
                if not np.isnan(ic):
                    daily_ics.append(ic)

            if daily_ics:
                mean_ic = np.mean(daily_ics)
                ic_ir = mean_ic / np.std(daily_ics) if np.std(daily_ics) > 0 else 0
                ic_results[feat] = {"mean_ic": mean_ic, "ic_ir": ic_ir}

        # 筛选
        selected = [f for f, v in ic_results.items() if abs(v["mean_ic"]) >= threshold]
        removed = [f for f, v in ic_results.items() if abs(v["mean_ic"]) < threshold]

        logger.info(
            "IC筛选 (threshold=%.3f): 保留 %d, 移除 %d",
            threshold, len(selected), len(removed)
        )

        if removed:
            logger.debug("移除的特征 (|IC|<%.3f): %s", threshold, removed)

        return selected

    # ==================== 相关性筛选 ====================

    def filter_by_correlation(
        self,
        features: pd.DataFrame,
        threshold: Optional[float] = None,
        date_col: str = "date",
    ) -> List[str]:
        """
        按相关性筛选特征

        基于相关系数矩阵，移除高相关性特征对中IC较低的一个。

        Args:
            features: 特征DataFrame
            threshold: 相关系数阈值, None则使用默认配置
            date_col: 日期列

        Returns:
            保留的特征列名列表
        """
        threshold = threshold if threshold is not None else self.corr_threshold

        feature_cols = [
            c for c in features.columns
            if c != date_col and pd.api.types.is_numeric_dtype(features[c])
        ]

        if len(feature_cols) <= 1:
            return feature_cols

        # 计算相关系数矩阵
        corr_matrix = features[feature_cols].corr().abs()

        # 找出高相关对并移除
        to_remove = set()
        cols = list(corr_matrix.columns)

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if cols[j] in to_remove:
                    continue
                if corr_matrix.iloc[i, j] >= threshold:
                    # 移除按名称排序靠后的（稳定顺序）
                    if cols[i] not in to_remove:
                        to_remove.add(cols[j])

        selected = [c for c in feature_cols if c not in to_remove]

        logger.info(
            "相关性筛选 (threshold=%.2f): 保留 %d, 移除 %d 个高相关特征",
            threshold, len(selected), len(to_remove)
        )

        if to_remove:
            logger.debug("移除的高相关特征: %s", sorted(to_remove)[:20])

        return selected

    def filter_by_ic_and_correlation(
        self,
        features: pd.DataFrame,
        labels: Union[pd.Series, pd.DataFrame],
        ic_threshold: Optional[float] = None,
        corr_threshold: Optional[float] = None,
        date_col: str = "date",
    ) -> List[str]:
        """
        组合IC筛选和相关性筛选

        先按IC筛选，再对保留特征做相关性去重。

        Args:
            features: 特征DataFrame
            labels: 标签
            ic_threshold: IC阈值
            corr_threshold: 相关性阈值
            date_col: 日期列

        Returns:
            最终保留的特征列名列表
        """
        logger.info("开始特征筛选 (IC + 相关性)...")

        # Step 1: IC 筛选
        ic_selected = self.filter_by_ic(features, labels, ic_threshold, date_col)
        logger.info("IC筛选后: %d 特征", len(ic_selected))

        # Step 2: 相关性筛选
        if len(ic_selected) > 1:
            features_subset = features[ic_selected]
            final_selected = self.filter_by_correlation(features_subset, corr_threshold, date_col)
            logger.info("相关性筛选后: %d 特征 (共移除 %d 个)", len(final_selected), len(ic_selected) - len(final_selected))
        else:
            final_selected = ic_selected

        logger.info("特征筛选完成: %d -> %d", len([c for c in features.columns if c != date_col and pd.api.types.is_numeric_dtype(features[c])]), len(final_selected))
        return final_selected


# ---- 模块级便捷函数 ----

def clean_features(df: pd.DataFrame, group_col: str = "code", date_col: str = "date") -> pd.DataFrame:
    """便捷清洗函数"""
    cleaner = DataCleaner()
    return cleaner.clean_features(df, group_col, date_col)


def winsorize(series: pd.Series, limits: Tuple[float, float] = (0.01, 0.01)) -> pd.Series:
    """便捷Winsorize函数"""
    cleaner = DataCleaner()
    return cleaner.winsorize(series, limits)


def standardize_cross_section(df: pd.DataFrame, group_col: str = "date", method: str = "zscore") -> pd.DataFrame:
    """便捷标准化函数"""
    cleaner = DataCleaner()
    return cleaner.standardize_cross_section(df, group_col, method)


def filter_by_ic(
    features: pd.DataFrame, labels: Union[pd.Series, pd.DataFrame],
    threshold: Optional[float] = None, date_col: str = "date",
) -> List[str]:
    """便捷IC筛选函数"""
    cleaner = DataCleaner()
    return cleaner.filter_by_ic(features, labels, threshold, date_col)


def filter_by_correlation(
    features: pd.DataFrame, threshold: Optional[float] = None, date_col: str = "date",
) -> List[str]:
    """便捷相关性筛选函数"""
    cleaner = DataCleaner()
    return cleaner.filter_by_correlation(features, threshold, date_col)