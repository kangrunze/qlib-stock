"""
横截面因子计算器 - CrossSectionCalculator

在每个交易日，对所有股票的因子进行横截面排名，生成相对强弱信号。
输入: Dict[str, pd.DataFrame] (股票代码 -> OHLCV DataFrame)
输出: Dict[str, pd.DataFrame] (股票代码 -> 横截面排名因子)

核心概念：
- rank: 排名百分比 (0-100)，数值越大排名越靠前
- pct: 分位数 (0-1)
- 排名按升序排列（原始值越小, rank 越小），如需降序排名则用原始值的负数

Author: stock-ai quantitative trading project
"""

import logging
from typing import Dict, Optional, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ==============================================================================
# 工具函数
# ==============================================================================

def _rank_pct(series: pd.Series, ascending: bool = True) -> pd.Series:
    """对序列计算排名百分比 (0-100)。

    使用 'average' 方法处理并列值。

    Args:
        series: 待排名序列
        ascending: True=升序排名（小值排名低），False=降序排名

    Returns:
        排名百分比序列 (0-100)
    """
    valid = series.notna()
    result = pd.Series(np.nan, index=series.index)
    if valid.sum() == 0:
        return result

    ranked = series[valid].rank(method="average", ascending=ascending, pct=False)
    # 转换为 0-100 的百分比排名
    n = len(ranked)
    if n > 1:
        result[valid] = (ranked - 1) / (n - 1) * 100
    else:
        result[valid] = 50.0
    return result


def _rank_quantile(series: pd.Series, ascending: bool = True) -> pd.Series:
    """对序列计算分位数 (0-1)。

    Args:
        series: 待排名序列
        ascending: True=升序，False=降序

    Returns:
        分位数序列 (0-1)
    """
    valid = series.notna()
    result = pd.Series(np.nan, index=series.index)
    if valid.sum() == 0:
        return result

    ranked = series[valid].rank(method="average", ascending=ascending, pct=False)
    n = len(ranked)
    if n > 1:
        result[valid] = (ranked - 1) / (n - 1)
    else:
        result[valid] = 0.5
    return result


def _safe_divide(a: np.ndarray, b: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
    """安全除法。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(a, b)
        result[~np.isfinite(result)] = fill_value
    return result


# ==============================================================================
# 主计算器类
# ==============================================================================

class CrossSectionCalculator:
    """横截面因子计算器。

    在所有股票之间，按交易日计算横截面排名因子。
    因子包括价格、成交量、成交额、换手率、振幅、MA偏离、
    市值代理、收益率、波动率、动量综合、RSI、MACD、
    成交量比率，以及行业相对排名（如果提供行业数据）。

    Usage:
        >>> cs = CrossSectionCalculator()
        >>> cs_factors = cs.compute_all(data_dict)
        >>> # 可选：传入行业数据
        >>> cs_factors = cs.compute_all(data_dict, industry_dict)
    """

    def __init__(self):
        """初始化横截面计算器。"""
        self._factor_names = []

    def compute_all(
        self,
        data_dict: Dict[str, pd.DataFrame],
        industry_dict: Optional[Dict[str, str]] = None,
        verbose: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """计算所有横截面因子。

        Args:
            data_dict: {股票代码: OHLCV DataFrame}，每个 DataFrame 必须包含
                       [open, high, low, close, volume] 列
            industry_dict: {股票代码: 行业名称}，可选
            verbose: 是否输出详细日志

        Returns:
            Dict[str, pd.DataFrame]，每个股票一个 DataFrame，
            列名为横截面因子名称，按日期索引

        Raises:
            ValueError: 如果 data_dict 为空
        """
        if not data_dict:
            raise ValueError("data_dict 为空")

        logger.info(
            "开始计算横截面因子，共 %d 只股票",
            len(data_dict),
        )

        # 步骤1: 为每只股票收集用于排名的每日数据
        daily_collections = self._collect_daily_data(data_dict, verbose=verbose)

        # 步骤2: 在每个交易日内计算横截面排名
        stock_cs_factors = self._compute_daily_ranks(
            daily_collections, data_dict, industry_dict, verbose=verbose
        )

        self._factor_names = list(daily_collections.keys())[:1]  # 标记
        logger.info("横截面因子计算完成")

        return stock_cs_factors

    def _collect_daily_data(
        self,
        data_dict: Dict[str, pd.DataFrame],
        verbose: bool = True,
    ) -> Dict[str, Dict[str, pd.Series]]:
        """为每只股票收集每日原始指标数据。

        收集的数据用于后续横截面对比。

        Args:
            data_dict: 股票数据字典
            verbose: 是否输出详细日志

        Returns:
            {指标名: {股票代码: 时间序列}}
        """
        if verbose:
            logger.info("收集每日原始指标数据...")

        # 定义要收集的指标及其计算方式
        collections = {}
        stock_dates = set()

        for stock, df in data_dict.items():
            df = df.sort_index()
            stock_dates.update(df.index)

            # 基础指标收集
            self._collect_basic_metrics(collections, stock, df)
            # 衍生指标收集
            self._collect_derived_metrics(collections, stock, df)

        return collections

    def _collect_basic_metrics(
        self,
        collections: dict,
        stock: str,
        df: pd.DataFrame,
    ):
        """收集基础指标（价格、成交量、成交额等）。

        Args:
            collections: 收集字典（原地修改）
            stock: 股票代码
            df: OHLCV DataFrame
        """
        # 收盘价
        self._add_to_collection(collections, "close", stock, df["close"])

        # 成交量
        self._add_to_collection(collections, "volume", stock, df["volume"])

        # 成交额
        if "amount" in df.columns:
            self._add_to_collection(collections, "amount", stock, df["amount"])

        # 换手率
        if "turnover_rate" in df.columns:
            self._add_to_collection(
                collections, "turnover", stock, df["turnover_rate"]
            )
        else:
            # 使用成交量代理
            turnover_proxy = df["volume"] / df["volume"].rolling(
                window=252, min_periods=60
            ).mean()
            self._add_to_collection(collections, "turnover", stock, turnover_proxy)

        # 振幅
        amplitude = (df["high"] - df["low"]) / df["close"].shift(1)
        self._add_to_collection(collections, "amplitude", stock, amplitude)

        # MA20 偏离度
        ma20 = df["close"].rolling(window=20, min_periods=20).mean()
        ma20_dev = _safe_divide(
            (df["close"] - ma20).values, ma20.values
        )
        self._add_to_collection(
            collections,
            "ma20_deviation",
            stock,
            pd.Series(ma20_dev, index=df.index),
        )

        # 市值代理 (close * volume)
        cap_proxy = df["close"] * df["volume"]
        self._add_to_collection(collections, "cap_proxy", stock, cap_proxy)

    def _collect_derived_metrics(
        self,
        collections: dict,
        stock: str,
        df: pd.DataFrame,
    ):
        """收集衍生指标（收益率、波动率、RSI、MACD 等）。

        Args:
            collections: 收集字典（原地修改）
            stock: 股票代码
            df: OHLCV DataFrame
        """
        close = df["close"]
        volume = df["volume"]

        # 5日收益率
        ret_5d = close.pct_change(periods=5) * 100
        self._add_to_collection(collections, "ret_5d", stock, ret_5d)

        # 20日收益率
        ret_20d = close.pct_change(periods=20) * 100
        self._add_to_collection(collections, "ret_20d", stock, ret_20d)

        # 20日波动率
        vol_20d = close.pct_change().rolling(window=20, min_periods=20).std()
        self._add_to_collection(collections, "volatility_20d", stock, vol_20d)

        # 动量综合得分 (Ret5d * 0.4 + Ret20d * 0.3 + RSI14_norm * 0.3)
        rsi14 = self._compute_rsi14_simple(close)
        mom_composite = (
            ret_5d.fillna(0) * 0.4
            + ret_20d.fillna(0) * 0.3
            + (rsi14.fillna(50) / 100.0 * 100) * 0.3
        )
        self._add_to_collection(collections, "momentum_composite", stock, mom_composite)

        # RSI14
        self._add_to_collection(collections, "rsi14", stock, rsi14)

        # MACD 得分
        macd_score = self._compute_macd_score_simple(close)
        self._add_to_collection(collections, "macd", stock, macd_score)

        # 成交量比率 (5日)
        vol_ratio_5d = _safe_divide(
            volume.values,
            volume.rolling(window=5, min_periods=5).mean().values,
        )
        self._add_to_collection(
            collections,
            "volume_ratio_5d",
            stock,
            pd.Series(vol_ratio_5d, index=df.index),
        )

    @staticmethod
    def _compute_rsi14_simple(close: pd.Series) -> pd.Series:
        """简化版 RSI14 计算。

        Args:
            close: 收盘价序列

        Returns:
            RSI14 序列
        """
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        rs = _safe_divide(avg_gain.values, avg_loss.values)
        return pd.Series(100.0 - 100.0 / (1.0 + rs), index=close.index)

    @staticmethod
    def _compute_macd_score_simple(close: pd.Series) -> pd.Series:
        """简化版 MACD 得分计算。

        Args:
            close: 收盘价序列

        Returns:
            MACD 得分序列 (DIF - DEA 归一化)
        """
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_hist = 2.0 * (dif - dea)
        # 用收盘价归一化
        macd_score = _safe_divide(macd_hist.values, close.values) * 100
        return pd.Series(macd_score, index=close.index)

    @staticmethod
    def _add_to_collection(
        collections: dict,
        metric_name: str,
        stock: str,
        series: pd.Series,
    ):
        """将股票指标数据添加到收集字典。

        Args:
            collections: 收集字典（原地修改）
            metric_name: 指标名称
            stock: 股票代码
            series: 指标时间序列
        """
        if metric_name not in collections:
            collections[metric_name] = {}
        collections[metric_name][stock] = series

    def _compute_daily_ranks(
        self,
        collections: dict,
        data_dict: Dict[str, pd.DataFrame],
        industry_dict: Optional[Dict[str, str]] = None,
        verbose: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """按交易日计算横截面排名。

        对每个交易日，汇总所有股票的同一指标值，
        进行横截面排名，生成 rank 和 pct 因子。

        Args:
            collections: 收集的每日指标数据
            data_dict: 原始股票数据字典
            industry_dict: 行业映射字典

        Returns:
            {股票代码: 横截面因子 DataFrame}
        """
        if verbose:
            logger.info("按交易日计算横截面排名...")

        # 初始化结果
        stock_cs_factors: Dict[str, pd.DataFrame] = {}
        for stock in data_dict:
            stock_cs_factors[stock] = pd.DataFrame(index=data_dict[stock].index)

        # 获取所有有数据的日期
        all_dates = self._get_common_dates(collections)

        # 对每个交易日计算横截面排名
        metric_names = list(collections.keys())
        total_dates = len(all_dates)

        for date_idx, date in enumerate(all_dates):
            if verbose and (date_idx % 250 == 0 or date_idx == total_dates - 1):
                logger.info(
                    "  处理横截面排名: %s (%d/%d)", date, date_idx + 1, total_dates
                )

            # 收集当日所有股票的指标值
            day_data: Dict[str, Dict[str, float]] = {m: {} for m in metric_names}

            for metric_name in metric_names:
                for stock, series in collections[metric_name].items():
                    if date in series.index:
                        val = series.loc[date]
                        if np.isfinite(val):
                            day_data[metric_name][stock] = val

            # 计算横截面排名
            self._compute_ranks_for_day(
                stock_cs_factors, day_data, date, industry_dict
            )

        return stock_cs_factors

    def _compute_ranks_for_day(
        self,
        stock_cs_factors: Dict[str, pd.DataFrame],
        day_data: Dict[str, Dict[str, float]],
        date,
        industry_dict: Optional[Dict[str, str]],
    ):
        """计算单个交易日的横截面排名因子。

        Args:
            stock_cs_factors: 结果字典（原地修改）
            day_data: {指标名: {股票代码: 值}}
            date: 当前日期
            industry_dict: 行业映射
        """
        # 获取当日有数据的股票列表
        stocks_with_data = set()
        for metric_data in day_data.values():
            stocks_with_data.update(metric_data.keys())

        if not stocks_with_data:
            return

        # 为每个指标计算排名
        rank_definitions = [
            ("close", "close_rank", "close_pct", True),
            ("volume", "volume_rank", "volume_pct", True),
            ("amount", "amount_rank", "amount_pct", True),
            ("turnover", "turnover_rank", "turnover_pct", True),
            ("amplitude", "amplitude_rank", "amplitude_pct", True),
            ("ma20_deviation", "ma20_deviation_rank", "ma20_deviation_pct", True),
            ("cap_proxy", "cap_proxy_rank", "cap_proxy_pct", True),
            ("ret_5d", "ret_5d_rank", "ret_5d_pct", True),
            ("ret_20d", "ret_20d_rank", "ret_20d_pct", True),
            ("volatility_20d", "volatility_20d_rank", "volatility_20d_pct", True),
            ("momentum_composite", "momentum_composite_rank", "momentum_composite_pct", True),
            ("rsi14", "rsi14_rank", "rsi14_pct", True),
            ("macd", "macd_rank", "macd_pct", True),
            ("volume_ratio_5d", "volume_ratio_5d_rank", "volume_ratio_5d_pct", True),
        ]

        for metric_name, rank_col, pct_col, ascending in rank_definitions:
            if metric_name not in day_data:
                continue
            stock_values = day_data[metric_name]
            if not stock_values:
                continue

            series = pd.Series(stock_values)
            rank_series = _rank_pct(series, ascending=ascending)
            pct_series = _rank_quantile(series, ascending=ascending)

            for stock, val in stock_values.items():
                if stock not in stock_cs_factors:
                    continue
                if date in stock_cs_factors[stock].index:
                    stock_cs_factors[stock].loc[date, rank_col] = rank_series.get(stock)
                    stock_cs_factors[stock].loc[date, pct_col] = pct_series.get(stock)

        # 行业相对排名（如果提供了行业数据）
        if industry_dict:
            self._compute_industry_relative_ranks(
                stock_cs_factors, day_data, date, industry_dict
            )

    def _compute_industry_relative_ranks(
        self,
        stock_cs_factors: Dict[str, pd.DataFrame],
        day_data: Dict[str, Dict[str, float]],
        date,
        industry_dict: Dict[str, str],
    ):
        """计算行业相对排名。

        在同一行业内对股票指标进行排名。

        Args:
            stock_cs_factors: 结果字典（原地修改）
            day_data: 当日指标数据
            date: 当前日期
            industry_dict: 行业映射
        """
        # 行业相对排名指标
        industry_metrics = [
            ("close", "close_industry_rank", "close_industry_pct"),
            ("ret_20d", "ret_20d_industry_rank", "ret_20d_industry_pct"),
            ("momentum_composite", "momentum_composite_industry_rank", "momentum_composite_industry_pct"),
            ("cap_proxy", "cap_proxy_industry_rank", "cap_proxy_industry_pct"),
        ]

        # 按行业分组
        industry_groups: Dict[str, list] = {}
        for stock, industry in industry_dict.items():
            if industry not in industry_groups:
                industry_groups[industry] = []
            industry_groups[industry].append(stock)

        for metric_name, rank_col, pct_col in industry_metrics:
            if metric_name not in day_data:
                continue

            stock_values = day_data[metric_name]

            for industry, stocks in industry_groups.items():
                # 获取该行业内有数据的股票
                ind_stocks = [s for s in stocks if s in stock_values]
                if len(ind_stocks) < 2:
                    continue

                ind_values = {s: stock_values[s] for s in ind_stocks}
                series = pd.Series(ind_values)
                rank_series = _rank_pct(series, ascending=True)
                pct_series = _rank_quantile(series, ascending=True)

                for stock in ind_stocks:
                    if stock not in stock_cs_factors:
                        continue
                    if date in stock_cs_factors[stock].index:
                        stock_cs_factors[stock].loc[date, rank_col] = rank_series.get(stock)
                        stock_cs_factors[stock].loc[date, pct_col] = pct_series.get(stock)

    @staticmethod
    def _get_common_dates(collections: dict) -> list:
        """获取所有股票有数据的公共日期列表。

        Args:
            collections: 收集的指标数据

        Returns:
            排序后的日期列表
        """
        all_dates = set()
        for metric_data in collections.values():
            for series in metric_data.values():
                all_dates.update(series.index)
        return sorted(all_dates)


# ==============================================================================
# 便捷入口
# ==============================================================================

def compute_cross_section_factors(
    data_dict: Dict[str, pd.DataFrame],
    industry_dict: Optional[Dict[str, str]] = None,
) -> Dict[str, pd.DataFrame]:
    """便捷函数：计算横截面因子。

    Args:
        data_dict: {股票代码: OHLCV DataFrame}
        industry_dict: {股票代码: 行业名称}，可选

    Returns:
        {股票代码: 横截面因子 DataFrame}
    """
    cs = CrossSectionCalculator()
    return cs.compute_all(data_dict, industry_dict)