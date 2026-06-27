"""
Alpha 经典因子计算器 - AlphaFactorCalculator

基于 WorldQuant 经典 Alpha 公式的因子计算器，使用纯 pandas/numpy 实现。
包含超过 15 个经典 Alpha 因子，涵盖动量、反转、相关性、波动率等信号。

输入: Dict[str, pd.DataFrame] (股票代码 -> OHLCV DataFrame)
输出: Dict[str, pd.DataFrame] (股票代码 -> Alpha 因子 DataFrame)

Author: stock-ai quantitative trading project
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ==============================================================================
# 工具函数
# ==============================================================================

def _rank(series: pd.Series) -> pd.Series:
    """对序列进行排名 (0-1 分位数)。

    Args:
        series: 输入序列

    Returns:
        排名序列 (0-1)
    """
    valid = series.notna()
    result = pd.Series(np.nan, index=series.index)
    if valid.sum() == 0:
        return result
    ranked = series[valid].rank(method="average", pct=False)
    n = len(ranked)
    if n > 1:
        result[valid] = (ranked - 1) / (n - 1)
    else:
        result[valid] = 0.5
    return result


def _ts_argmax(series: pd.Series, window: int) -> pd.Series:
    """时间序列滚动窗口内的最大值索引。

    Args:
        series: 输入序列
        window: 滚动窗口大小

    Returns:
        每个窗口内最大值的位置索引 (0-based)
    """
    def _argmax(x):
        x = x[~np.isnan(x)]
        if len(x) == 0:
            return np.nan
        return np.argmax(x)
    return series.rolling(window=window, min_periods=window).apply(_argmax, raw=True)


def _ts_min(series: pd.Series, window: int) -> pd.Series:
    """时间序列滚动窗口内的最小值。

    Args:
        series: 输入序列
        window: 滚动窗口大小

    Returns:
        滚动最小值序列
    """
    return series.rolling(window=window, min_periods=window).min()


def _ts_max(series: pd.Series, window: int) -> pd.Series:
    """时间序列滚动窗口内的最大值。

    Args:
        series: 输入序列
        window: 滚动窗口大小

    Returns:
        滚动最大值序列
    """
    return series.rolling(window=window, min_periods=window).max()


def _ts_sum(series: pd.Series, window: int) -> pd.Series:
    """时间序列滚动窗口内的求和。

    Args:
        series: 输入序列
        window: 滚动窗口大小

    Returns:
        滚动求和序列
    """
    return series.rolling(window=window, min_periods=window).sum()


def _correlation(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """滚动相关系数。

    Args:
        x: 第一个序列
        y: 第二个序列
        window: 滚动窗口

    Returns:
        滚动相关系数序列
    """
    return x.rolling(window=window, min_periods=window).corr(y)


def _covariance(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """滚动协方差。

    Args:
        x: 第一个序列
        y: 第二个序列
        window: 滚动窗口

    Returns:
        滚动协方差序列
    """
    return x.rolling(window=window, min_periods=window).cov(y)


def _scale(series: pd.Series, scale_factor: float = 1.0) -> pd.Series:
    """缩放序列使其总和为 scale_factor。

    Args:
        series: 输入序列
        scale_factor: 缩放目标总和

    Returns:
        缩放后序列
    """
    s = series.abs().sum()
    if s > 0 and np.isfinite(s):
        return series / s * scale_factor
    return series


def _signed_power(series: pd.Series, exponent: float) -> pd.Series:
    """带符号的幂运算: sign(x) * |x|^exponent。

    Args:
        series: 输入序列
        exponent: 指数

    Returns:
        带符号幂序列
    """
    return np.sign(series) * (np.abs(series) ** exponent)


def _delta(series: pd.Series, period: int = 1) -> pd.Series:
    """差分运算: series - series.shift(period)。

    Args:
        series: 输入序列
        period: 差分周期

    Returns:
        差分序列
    """
    return series - series.shift(period)


def _delay(series: pd.Series, period: int) -> pd.Series:
    """延迟运算: series.shift(period)。

    Args:
        series: 输入序列
        period: 延迟周期

    Returns:
        延迟序列
    """
    return series.shift(period)


def _safe_divide(a: np.ndarray, b: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
    """安全除法。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(a, b)
        result[~np.isfinite(result)] = fill_value
    return result


def _returns(close: pd.Series) -> pd.Series:
    """日收益率。"""
    return close.pct_change()


def _stddev(series: pd.Series, window: int) -> pd.Series:
    """滚动标准差。"""
    return series.rolling(window=window, min_periods=window).std()


# ==============================================================================
# Alpha 因子计算函数
# ==============================================================================

def alpha001(df: pd.DataFrame) -> pd.Series:
    """Alpha001: (rank(TsArgMax(SignedPower(returns, 2), 5)) - 0.5)。

    寻找过去5天内收益率平方最大的那一天的位置。

    Args:
        df: OHLCV DataFrame

    Returns:
        Alpha001 序列
    """
    ret = _returns(df["close"])
    power = _signed_power(ret, 2.0)
    argmax = _ts_argmax(power, 5)
    ranked = _rank(argmax)
    return ranked - 0.5


def alpha002(df: pd.DataFrame) -> pd.Series:
    """Alpha002: -1 * correlation(rank(delta(log(volume), 2)), rank((close-open)/open), 6)。

    成交量变化率与日内收益率的相关性取反。

    Args:
        df: OHLCV DataFrame

    Returns:
        Alpha002 序列
    """
    volume = df["volume"].replace(0, np.nan)
    delta_log_vol = _delta(np.log(volume), 2)
    intraday_ret = (df["close"] - df["open"]) / df["open"]
    corr = _correlation(_rank(delta_log_vol), _rank(intraday_ret), 6)
    return -1.0 * corr


def alpha003(df: pd.DataFrame) -> pd.Series:
    """Alpha003: -1 * correlation(rank(open), rank(volume), 10)。

    开盘价排名与成交量排名的负相关性。

    Args:
        df: OHLCV DataFrame

    Returns:
        Alpha003 序列
    """
    return -1.0 * _correlation(_rank(df["open"]), _rank(df["volume"]), 10)


def alpha004(df: pd.DataFrame) -> pd.Series:
    """Alpha004: -1 * TsRank(rank(low), 9)。低价的滚动排名（反转信号）。

    Args:
        df: OHLCV DataFrame

    Returns:
        Alpha004 序列
    """
    low_rank = _rank(df["low"])
    # 滚动排名：在窗口内再排名
    def _rolling_rank(x):
        x = x[~np.isnan(x)]
        if len(x) == 0:
            return np.nan
        return (pd.Series(x).rank().iloc[-1] - 1) / (len(x) - 1) if len(x) > 1 else 0.5
    ts_rank = low_rank.rolling(window=9, min_periods=9).apply(_rolling_rank, raw=True)
    return -1.0 * ts_rank


def alpha005(df: pd.DataFrame) -> pd.Series:
    """Alpha005: rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap))))。

    开盘价与10日均VWAP偏离度排名, 结合收盘价与VWAP偏离度。

    Args:
        df: OHLCV DataFrame

    Returns:
        Alpha005 序列
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vwap_daily = tp  # 简化为 TP
    vwap_ma10 = _ts_sum(vwap_daily, 10) / 10.0
    a = _rank(df["open"] - vwap_ma10)
    b = -1.0 * np.abs(_rank(df["close"] - vwap_daily))
    return a * b


def alpha006(df: pd.DataFrame) -> pd.Series:
    """Alpha006: -1 * correlation(open, volume, 10)。

    开盘价与成交量的负相关性。

    Args:
        df: OHLCV DataFrame

    Returns:
        Alpha006 序列
    """
    return -1.0 * _correlation(df["open"], df["volume"], 10)


def alpha009(df: pd.DataFrame) -> pd.Series:
    """Alpha009: RSI-based factor - (close - ts_min(low, 5)) / (ts_max(high, 5) - ts_min(low, 5))。

    基于价格在5日范围内的位置，类似于随机指标。

    Args:
        df: OHLCV DataFrame

    Returns:
        Alpha009 序列
    """
    low_5 = _ts_min(df["low"], 5)
    high_5 = _ts_max(df["high"], 5)
    result = _safe_divide(
        df["close"].values - low_5.values,
        high_5.values - low_5.values,
    )
    return pd.Series(result, index=df.index)


def alpha012(df: pd.DataFrame) -> pd.Series:
    """Alpha012: sign(delta(volume, 1)) * (-1 * delta(close, 1))。

    成交量变化方向与价格变化方向的交互。

    Args:
        df: OHLCV DataFrame

    Returns:
        Alpha012 序列
    """
    return np.sign(_delta(df["volume"], 1)) * (-1.0 * _delta(df["close"], 1))


def alpha017(df: pd.DataFrame) -> pd.Series:
    """Alpha017: -1 * rank(stddev(returns, 20)) * correlation(close, volume, 20)。

    波动率排名与价量相关性的乘积取反。

    Args:
        df: OHLCV DataFrame

    Returns:
        Alpha017 序列
    """
    std_ret = _stddev(_returns(df["close"]), 20)
    corr = _correlation(df["close"], df["volume"], 20)
    return -1.0 * _rank(std_ret) * corr


def alpha028(df: pd.DataFrame) -> pd.Series:
    """Alpha028: correlation(adv20, low, 20) + (high + low) / 2 - close。

    ADV20与低价的相关性 + 中点价格与收盘价的偏离。

    Args:
        df: OHLCV DataFrame

    Returns:
        Alpha028 序列
    """
    adv20 = _ts_sum(df["volume"], 20) / 20.0
    corr = _correlation(adv20, df["low"], 20)
    mid_price = (df["high"] + df["low"]) / 2.0
    return corr + (mid_price - df["close"])


def alpha032(df: pd.DataFrame) -> pd.Series:
    """Alpha032: scale of (correlation(close, volume, 20) * close)。

    价量相关性缩放乘收盘价。

    Args:
        df: OHLCV DataFrame

    Returns:
        Alpha032 序列
    """
    corr = _correlation(df["close"], df["volume"], 20)
    return _scale(corr * df["close"])


def alpha035(df: pd.DataFrame) -> pd.Series:
    """Alpha035: 动量与成交量组合因子。

    综合 Ret5d + Ret20d + VolRatio5d 的多信号组合。

    Args:
        df: OHLCV DataFrame

    Returns:
        Alpha035 序列
    """
    ret_5d = _returns(df["close"]).rolling(5).sum()
    ret_20d = _returns(df["close"]).rolling(20).sum()
    vol_ratio = _safe_divide(
        df["volume"].values,
        _ts_sum(df["volume"], 5).values,
    )
    ret_5d_rank = _rank(ret_5d)
    ret_20d_rank = _rank(ret_20d)
    vol_ratio_series = pd.Series(vol_ratio, index=df.index)
    vol_ratio_rank = _rank(vol_ratio_series)
    return (ret_5d_rank * 0.3 + ret_20d_rank * 0.4 + vol_ratio_rank * 0.3)


def alpha049(df: pd.DataFrame) -> pd.Series:
    """Alpha049: -1 * delta(close, 10)。

    简单的10日价格反转信号。

    Args:
        df: OHLCV DataFrame

    Returns:
        Alpha049 序列
    """
    return -1.0 * _delta(df["close"], 10)


def alpha053(df: pd.DataFrame) -> pd.Series:
    """Alpha053: 价格在20日范围内的位置。

    (close - ts_min(low, 20)) / (ts_max(high, 20) - ts_min(low, 20)) - 0.5

    Args:
        df: OHLCV DataFrame

    Returns:
        Alpha053 序列
    """
    low_20 = _ts_min(df["low"], 20)
    high_20 = _ts_max(df["high"], 20)
    position = _safe_divide(
        df["close"].values - low_20.values,
        high_20.values - low_20.values,
    )
    return pd.Series(position, index=df.index) - 0.5


def alpha060(df: pd.DataFrame) -> pd.Series:
    """Alpha060: -1 * (close - vwap20) / std(close, 20)。

    价格相对于20日均价的偏离度（标准化反转）。

    Args:
        df: OHLCV DataFrame

    Returns:
        Alpha060 序列
    """
    vwap20 = df["close"].rolling(window=20, min_periods=20).mean()
    std20 = _stddev(df["close"], 20)
    deviation = _safe_divide(
        df["close"].values - vwap20.values,
        std20.values,
    )
    return pd.Series(-1.0 * deviation, index=df.index)


def alpha083(df: pd.DataFrame) -> pd.Series:
    """Alpha083: rank(delta(vwap, 1)) * rank(correlation(close, volume, 5))。

    VWAP变化排名与短期价量相关性排名的乘积。

    Args:
        df: OHLCV DataFrame

    Returns:
        Alpha083 序列
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vwap_delta = _delta(tp, 1)
    rank_delta = _rank(vwap_delta)
    corr = _correlation(df["close"], df["volume"], 5)
    rank_corr = _rank(corr)
    return rank_delta * rank_corr


def alpha101_inspired(df: pd.DataFrame) -> pd.Series:
    """Alpha101 启发因子: rank(correlation(ret, volume_ret, period))。

    收益率与成交量变化率的相关性排名。

    Args:
        df: OHLCV DataFrame

    Returns:
        Alpha101 序列
    """
    ret = _returns(df["close"])
    volume_ret = _returns(df["volume"])
    corr_20 = _correlation(ret, volume_ret, 20)
    return _rank(corr_20)


def alpha_bb_squeeze(df: pd.DataFrame) -> pd.Series:
    """Alpha BB squeeze: 布林带宽度压缩因子。

    布林带宽度在过去N日的排名，宽度越小分数越高（预示突破）。

    Args:
        df: OHLCV DataFrame

    Returns:
        BB Squeeze 序列
    """
    close = df["close"]
    mid = close.rolling(window=20, min_periods=20).mean()
    std = close.rolling(window=20, min_periods=20).std()
    upper = mid + 2.0 * std
    lower = mid - 2.0 * std
    bb_width = _safe_divide(upper.values - lower.values, mid.values)
    bb_width_series = pd.Series(bb_width, index=df.index)
    # 宽度压缩 -> 分数高（取反排名）
    ranked = _rank(bb_width_series)
    return 1.0 - ranked  # 压缩时分数高


def alpha_volatility_breakout(df: pd.DataFrame) -> pd.Series:
    """Alpha 波动率突破: 当日收益率/20日波动率。

    突破强度因子。

    Args:
        df: OHLCV DataFrame

    Returns:
        波动率突破序列
    """
    ret = _returns(df["close"])
    vol20 = _stddev(df["close"], 20)
    return _safe_divide(ret.values, vol20.values)


# ==============================================================================
# 因子注册表 (所有可计算的 Alpha 因子)
# ==============================================================================

ALPHA_REGISTRY = {
    "Alpha001": alpha001,
    "Alpha002": alpha002,
    "Alpha003": alpha003,
    "Alpha004": alpha004,
    "Alpha005": alpha005,
    "Alpha006": alpha006,
    "Alpha009": alpha009,
    "Alpha012": alpha012,
    "Alpha017": alpha017,
    "Alpha028": alpha028,
    "Alpha032": alpha032,
    "Alpha035": alpha035,
    "Alpha049": alpha049,
    "Alpha053": alpha053,
    "Alpha060": alpha060,
    "Alpha083": alpha083,
    "Alpha101": alpha101_inspired,
    "AlphaBB": alpha_bb_squeeze,
    "AlphaBreakout": alpha_volatility_breakout,
}


# ==============================================================================
# 主计算器类
# ==============================================================================

class AlphaFactorCalculator:
    """Alpha 经典因子计算器。

    基于 WorldQuant 风格公式计算 Alpha 因子。
    支持所有注册表中的因子，可选择性计算。

    Usage:
        >>> calc = AlphaFactorCalculator()
        >>> alpha_factors = calc.compute_all(data_dict)
        >>> # 只计算特定因子
        >>> alpha_factors = calc.compute_all(data_dict, alpha_names=["Alpha001", "Alpha012"])
    """

    def __init__(self, alpha_names: list = None):
        """初始化 Alpha 因子计算器。

        Args:
            alpha_names: 要计算的因子名称列表。如果为 None，计算所有已注册因子。
        """
        if alpha_names is None:
            alpha_names = list(ALPHA_REGISTRY.keys())
        # 过滤有效的因子名称
        self.alpha_names = [n for n in alpha_names if n in ALPHA_REGISTRY]
        invalid = set(alpha_names) - set(ALPHA_REGISTRY.keys())
        if invalid:
            logger.warning("无效的 Alpha 因子名称: %s", invalid)
        self._computed_factors = []

    def compute_all(
        self,
        data_dict: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.DataFrame]:
        """为所有股票计算 Alpha 因子。

        Args:
            data_dict: {股票代码: OHLCV DataFrame}

        Returns:
            {股票代码: Alpha 因子 DataFrame}

        Raises:
            ValueError: 如果 data_dict 为空或缺少必要列
        """
        if not data_dict:
            raise ValueError("data_dict 为空")

        logger.info(
            "开始计算 Alpha 因子: 共 %d 只股票, %d 个因子",
            len(data_dict),
            len(self.alpha_names),
        )

        result = {}
        total = len(data_dict)

        for idx, (stock, df) in enumerate(data_dict.items()):
            if df.empty:
                logger.warning("股票 %s 的数据为空，跳过", stock)
                result[stock] = pd.DataFrame()
                continue

            # 验证必要列
            required = ["open", "high", "low", "close", "volume"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                logger.warning("股票 %s 缺少列 %s，跳过", stock, missing)
                result[stock] = pd.DataFrame()
                continue

            df = df.sort_index().copy()
            df = df.replace([np.inf, -np.inf], np.nan)

            stock_factors = self._compute_stock_alphas(df, stock)
            result[stock] = stock_factors

            if (idx + 1) % 200 == 0 or idx == total - 1:
                logger.info(
                    "  Alpha 因子进度: %d/%d 只股票",
                    idx + 1,
                    total,
                )

        self._computed_factors = self.alpha_names
        logger.info("Alpha 因子计算完成，共计算 %d 个因子", len(self.alpha_names))
        return result

    def _compute_stock_alphas(
        self,
        df: pd.DataFrame,
        stock: str,
    ) -> pd.DataFrame:
        """为单只股票计算所有选定的 Alpha 因子。

        Args:
            df: OHLCV DataFrame
            stock: 股票代码

        Returns:
            Alpha 因子 DataFrame
        """
        factors = pd.DataFrame(index=df.index)

        for alpha_name in self.alpha_names:
            try:
                alpha_func = ALPHA_REGISTRY[alpha_name]
                factor = alpha_func(df)
                # 确保返回的是 Series
                if isinstance(factor, pd.DataFrame) and factor.shape[1] == 1:
                    factor = factor.iloc[:, 0]
                if isinstance(factor, np.ndarray):
                    factor = pd.Series(factor, index=df.index)
                factors[alpha_name] = factor
            except Exception as e:
                logger.warning(
                    "股票 %s 的 %s 计算失败: %s",
                    stock,
                    alpha_name,
                    e,
                )
                factors[alpha_name] = np.nan

        # 清理
        factors = factors.replace([np.inf, -np.inf], np.nan)

        return factors

    @property
    def computed_factors(self) -> list:
        """返回已计算的因子名称列表。"""
        return self._computed_factors

    @property
    def factor_count(self) -> int:
        """返回已计算因子数量。"""
        return len(self._computed_factors)


# ==============================================================================
# 便捷入口
# ==============================================================================

def compute_alpha_factors(
    data_dict: Dict[str, pd.DataFrame],
    alpha_names: list = None,
) -> Dict[str, pd.DataFrame]:
    """便捷函数：计算 Alpha 因子。

    Args:
        data_dict: {股票代码: OHLCV DataFrame}
        alpha_names: 要计算的因子名称列表

    Returns:
        {股票代码: Alpha 因子 DataFrame}
    """
    calc = AlphaFactorCalculator(alpha_names=alpha_names)
    return calc.compute_all(data_dict)