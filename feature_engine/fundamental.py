"""
基本面因子计算器 - FundamentalFactorCalculator

基于每日交易数据构建基本面因子的代理变量。
所有因子均为近似值，使用 close*volume 作为市值代理，
并从价格/成交量变化中推导营收/利润等基本面特征。

输入的财务数据（如果可用）会优先使用；
否则全部从 OHLCV 数据推导代理变量。

输入: Dict[str, pd.DataFrame] (股票代码 -> OHLCV DataFrame)
输出: Dict[str, pd.DataFrame] (股票代码 -> 基本面因子 DataFrame)

Author: stock-ai quantitative trading project
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config.settings import SettingsManager

logger = logging.getLogger(__name__)


# ==============================================================================
# 工具函数
# ==============================================================================

def _safe_divide(a: np.ndarray, b: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
    """安全除法，避免除以零或无穷值。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(a, b)
        result[~np.isfinite(result)] = fill_value
    return result


def _rolling_sum(series: pd.Series, window: int) -> pd.Series:
    """滚动窗口求和。

    Args:
        series: 输入序列
        window: 窗口大小

    Returns:
        滚动求和序列
    """
    return series.rolling(window=window, min_periods=max(1, window // 2)).sum()


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """滚动窗口均值。

    Args:
        series: 输入序列
        window: 窗口大小

    Returns:
        滚动均值序列
    """
    return series.rolling(window=window, min_periods=max(1, window // 2)).mean()


# ==============================================================================
# 主计算器类
# ==============================================================================

class FundamentalFactorCalculator:
    """基本面因子计算器。

    基于 OHLCV 数据计算基本面因子代理变量。
    如有真实财务数据（EPS、股东权益、净利润等），优先使用；
    否则从价格和成交量推导代理变量。

    计算因子：
        - market_cap_proxy: 市值代理 (close * volume)
        - PE_proxy: 市盈率代理
        - PB_proxy: 市净率代理
        - ROE_proxy: ROE 代理
        - revenue_growth_proxy: 营收增长率代理
        - profit_margin_proxy: 利润率代理
        - debt_ratio_proxy: 负债率代理
        - price_to_book_proxy: 市净率代理（重复，统一命名）
        - earnings_yield_proxy: 盈利收益率代理 (1/PE)
        - market_cap_log: 市值代理的对数
        - turnover_proxy: 换手率代理（基于成交量标准化）

    Usage:
        >>> calc = FundamentalFactorCalculator()
        >>> fundamental_factors = calc.compute_all(data_dict)
        >>> # 传入真实财务数据
        >>> fundamental_factors = calc.compute_all(
        ...     data_dict,
        ...     financial_data=financial_dict,
        ... )
    """

    def __init__(self):
        """初始化基本面因子计算器。"""
        self.settings = SettingsManager()
        self._computed_factors = []

    def compute_all(
        self,
        data_dict: Dict[str, pd.DataFrame],
        financial_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """为所有股票计算基本面因子。

        Args:
            data_dict: {股票代码: OHLCV DataFrame}
            financial_data: {股票代码: 财务数据 DataFrame}，可选。
                            财务数据应包含 eps, book_value, net_profit,
                            revenue, total_assets, total_liabilities 列
                            （或任何可用的列）。

        Returns:
            {股票代码: 基本面因子 DataFrame}

        Raises:
            ValueError: 如果 data_dict 为空
        """
        if not data_dict:
            raise ValueError("data_dict 为空")

        logger.info(
            "开始计算基本面因子: 共 %d 只股票, 财务数据%s",
            len(data_dict),
            "已提供" if financial_data else "未提供（使用代理）",
        )

        result = {}
        total = len(data_dict)

        for idx, (stock, df) in enumerate(data_dict.items()):
            if df.empty:
                logger.warning("股票 %s 的数据为空，跳过", stock)
                result[stock] = pd.DataFrame()
                continue

            df = df.sort_index().copy()
            df = df.replace([np.inf, -np.inf], np.nan)

            # 获取该股票的财务数据（如果有）
            fin_df = None
            if financial_data and stock in financial_data:
                fin_df = financial_data[stock]

            # 计算基本面因子
            stock_factors = self._compute_stock_fundamentals(df, fin_df, stock)
            result[stock] = stock_factors

            if (idx + 1) % 200 == 0 or idx == total - 1:
                logger.info(
                    "  基本面因子进度: %d/%d 只股票",
                    idx + 1,
                    total,
                )

        if result:
            first_stock = list(result.keys())[0]
            self._computed_factors = list(result[first_stock].columns)

        logger.info(
            "基本面因子计算完成，共计算 %d 个因子",
            len(self._computed_factors),
        )
        return result

    def _compute_stock_fundamentals(
        self,
        df: pd.DataFrame,
        fin_df: Optional[pd.DataFrame],
        stock: str,
    ) -> pd.DataFrame:
        """为单只股票计算基本面因子。

        Args:
            df: OHLCV DataFrame
            fin_df: 财务数据 DataFrame 或 None
            stock: 股票代码

        Returns:
            基本面因子 DataFrame
        """
        factors = pd.DataFrame(index=df.index)
        close = df["close"]
        volume = df["volume"]
        amount = df.get("amount", df["close"] * df["volume"])

        # ---- 市值代理 (close * volume) ----
        cap_proxy = close * volume
        factors["market_cap_proxy"] = cap_proxy
        factors["market_cap_log"] = np.log(cap_proxy.replace(0, np.nan))

        # ---- 换手率代理 ----
        # 使用 volume / 252日平均volume 作为换手率代理
        avg_vol_252 = _rolling_mean(volume, 252)
        turnover_proxy = _safe_divide(volume.values, avg_vol_252.values)
        factors["turnover_proxy"] = turnover_proxy

        # ---- PE 代理 ----
        factors["PE_proxy"] = self._compute_pe_proxy(close, volume, fin_df, amount)

        # ---- PB 代理 ----
        factors["PB_proxy"] = self._compute_pb_proxy(close, volume, fin_df)

        # ---- ROE 代理 ----
        factors["ROE_proxy"] = self._compute_roe_proxy(close, volume, fin_df)

        # ---- 营收增长率代理 ----
        factors["revenue_growth_proxy"] = self._compute_revenue_growth(df, amount, fin_df)

        # ---- 利润率代理 ----
        factors["profit_margin_proxy"] = self._compute_profit_margin(df, amount, cap_proxy, fin_df)

        # ---- 负债率代理 ----
        factors["debt_ratio_proxy"] = self._compute_debt_ratio(fin_df)

        # ---- 盈利收益率代理 ----
        pe_proxy_arr = factors["PE_proxy"].values
        factors["earnings_yield_proxy"] = _safe_divide(
            np.ones(len(pe_proxy_arr)), pe_proxy_arr
        )

        # 清理
        factors = factors.replace([np.inf, -np.inf], np.nan)

        return factors

    def _compute_pe_proxy(
        self,
        close: pd.Series,
        volume: pd.Series,
        fin_df: Optional[pd.DataFrame],
        amount: pd.Series,
    ) -> np.ndarray:
        """计算 PE 代理。

        PE = 市值 / 净利润
        有财务数据时: PE = market_cap / net_profit
        无财务数据时: PE = market_cap / (amount 年化代理)

        Args:
            close: 收盘价
            volume: 成交量
            fin_df: 财务数据
            amount: 成交额

        Returns:
            PE 代理数组
        """
        cap_proxy = close * volume

        if fin_df is not None and "net_profit" in fin_df.columns:
            # 使用真实净利润（季度或年度），映射到日线
            net_profit = fin_df["net_profit"].resample("D").ffill().reindex(
                df_index=close.index, method="ffill"
            )
            return _safe_divide(cap_proxy.values, net_profit.values)
        else:
            # 代理: 使用成交额的年化值作为净利润代理
            # 年化成交额 (252日平均 × 252)
            annual_amount = _rolling_mean(amount, 252) * 252
            pe = _safe_divide(cap_proxy.values, annual_amount.values)
            # 限制 PE 范围在合理区间
            pe = np.clip(pe, 0, 1000)
            return pe

    def _compute_pb_proxy(
        self,
        close: pd.Series,
        volume: pd.Series,
        fin_df: Optional[pd.DataFrame],
    ) -> np.ndarray:
        """计算 PB 代理。

        PB = 市值 / 净资产
        有财务数据时: PB = market_cap / book_value
        无财务数据时: 使用 close / (close的60日均值) 作为近似

        Args:
            close: 收盘价
            volume: 成交量
            fin_df: 财务数据

        Returns:
            PB 代理数组
        """
        cap_proxy = close * volume

        if fin_df is not None and "book_value" in fin_df.columns:
            book_value = fin_df["book_value"].resample("D").ffill().reindex(
                df_index=close.index, method="ffill"
            )
            return _safe_divide(cap_proxy.values, book_value.values)
        else:
            # 代理: price / 60日均价 (简化的 PB 代理)
            avg_price = _rolling_mean(close, 60)
            pb_proxy = _safe_divide(close.values, avg_price.values)
            return pb_proxy

    def _compute_roe_proxy(
        self,
        close: pd.Series,
        volume: pd.Series,
        fin_df: Optional[pd.DataFrame],
    ) -> np.ndarray:
        """计算 ROE 代理。

        ROE = 净利润 / 净资产
        无财务数据时使用价格动量代理。

        Args:
            close: 收盘价
            volume: 成交量
            fin_df: 财务数据

        Returns:
            ROE 代理数组
        """
        if fin_df is not None and "net_profit" in fin_df.columns and "book_value" in fin_df.columns:
            net_profit = fin_df["net_profit"].resample("D").ffill().reindex(
                df_index=close.index, method="ffill"
            )
            book_value = fin_df["book_value"].resample("D").ffill().reindex(
                df_index=close.index, method="ffill"
            )
            return _safe_divide(net_profit.values, book_value.values)
        else:
            # 代理: 60日年化收益率
            ret_60d = close.pct_change(60)
            annualized = ret_60d * (252.0 / 60.0)
            # 限制在合理范围
            roe_proxy = np.clip(annualized.values, -1.0, 2.0)
            return roe_proxy

    def _compute_revenue_growth(
        self,
        df: pd.DataFrame,
        amount: pd.Series,
        fin_df: Optional[pd.DataFrame],
    ) -> np.ndarray:
        """计算营收增长率代理。

        有财务数据时: revenue YoY 增长率
        无财务数据时: 成交额的 20日 vs 60日 增长率差

        Args:
            df: OHLCV DataFrame
            amount: 成交额
            fin_df: 财务数据

        Returns:
            营收增长率代理数组
        """
        if fin_df is not None and "revenue" in fin_df.columns:
            revenue = fin_df["revenue"].resample("D").ffill().reindex(
                df_index=df.index, method="ffill"
            )
            # 年化增长率: 252日 diff / 252日前值
            rev_growth = _safe_divide(
                (revenue - revenue.shift(252)).values,
                revenue.shift(252).values,
            )
            return rev_growth
        else:
            # 代理: 成交额的近期增长 vs 长期增长
            amount_20d = _rolling_mean(amount, 20)
            amount_60d = _rolling_mean(amount, 60)
            growth = _safe_divide(
                (amount_20d - amount_60d).values,
                amount_60d.values,
            )
            return growth

    def _compute_profit_margin(
        self,
        df: pd.DataFrame,
        amount: pd.Series,
        cap_proxy: pd.Series,
        fin_df: Optional[pd.DataFrame],
    ) -> np.ndarray:
        """计算利润率代理。

        有财务数据时: net_profit / revenue
        无财务数据时: (close变化 / cap_proxy) 代理

        Args:
            df: OHLCV DataFrame
            amount: 成交额
            cap_proxy: 市值代理
            fin_df: 财务数据

        Returns:
            利润率代理数组
        """
        if fin_df is not None and "net_profit" in fin_df.columns and "revenue" in fin_df.columns:
            net_profit = fin_df["net_profit"].resample("D").ffill().reindex(
                df_index=df.index, method="ffill"
            )
            revenue = fin_df["revenue"].resample("D").ffill().reindex(
                df_index=df.index, method="ffill"
            )
            return _safe_divide(net_profit.values, revenue.values)
        else:
            # 代理: amount / market_cap 作为周转率代理（类利润率）
            margin_proxy = _safe_divide(amount.values, cap_proxy.values)
            return margin_proxy

    def _compute_debt_ratio(
        self,
        fin_df: Optional[pd.DataFrame],
    ) -> float:
        """计算负债率代理。

        有财务数据时: total_liabilities / total_assets
        无财务数据时: 默认 0.3 (中性假设)

        Args:
            fin_df: 财务数据

        Returns:
            负债率代理（单个值或默认值）
        """
        if fin_df is not None and "total_liabilities" in fin_df.columns and "total_assets" in fin_df.columns:
            liab = fin_df["total_liabilities"].mean()
            assets = fin_df["total_assets"].mean()
            if assets and assets > 0:
                return liab / assets
        return 0.3

    @property
    def computed_factors(self) -> list:
        """返回已计算的因子名称列表。"""
        return self._computed_factors

    @property
    def factor_count(self) -> int:
        """返回因子数量。"""
        return len(self._computed_factors)


# ==============================================================================
# 便捷入口
# ==============================================================================

def compute_fundamental_factors(
    data_dict: Dict[str, pd.DataFrame],
    financial_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, pd.DataFrame]:
    """便捷函数：计算基本面因子。

    Args:
        data_dict: {股票代码: OHLCV DataFrame}
        financial_data: {股票代码: 财务数据 DataFrame}，可选

    Returns:
        {股票代码: 基本面因子 DataFrame}
    """
    calc = FundamentalFactorCalculator()
    return calc.compute_all(data_dict, financial_data)