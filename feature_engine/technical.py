"""
技术指标计算器 - TechnicalFactorCalculator

基于日线 OHLCV 数据计算 100+ 技术因子，涵盖趋势、动量、波动率、
成交量/价格、统计五大类指标。全部使用 numpy/pandas 原生实现，
不依赖 ta-lib 等第三方技术分析库。

输入: DataFrame with columns [open, high, low, close, volume, amount]
输出: DataFrame indexed by date with all computed factors

Author: stock-ai quantitative trading project
"""

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ==============================================================================
# 工具函数
# ==============================================================================

def _safe_divide(a: np.ndarray, b: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
    """安全除法，避免除以零。
    
    Args:
        a: 分子数组
        b: 分母数组
        fill_value: 分母为零时的填充值
        
    Returns:
        除法结果
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(a, b)
        result[~np.isfinite(result)] = fill_value
    return result


def _rolling_apply(series: pd.Series, window: int, func, min_periods: int = None) -> pd.Series:
    """对序列应用滚动窗口函数，统一处理 NaN。
    
    Args:
        series: 输入序列
        window: 窗口大小
        func: 应用函数
        min_periods: 最小观测数，默认等于 window
        
    Returns:
        滚动计算结果
    """
    if min_periods is None:
        min_periods = max(1, window // 2)
    return series.rolling(window=window, min_periods=min_periods).apply(
        func, raw=True
    )


def _ema(series: pd.Series, span: int) -> pd.Series:
    """指数移动平均。
    
    Args:
        series: 输入序列
        span: EMA 跨度
        
    Returns:
        EMA 序列
    """
    return series.ewm(span=span, adjust=False).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    """简单移动平均。
    
    Args:
        series: 输入序列
        window: 窗口大小
        
    Returns:
        SMA 序列
    """
    return series.rolling(window=window, min_periods=1).mean()


def _wilder_smoothing(series: pd.Series, period: int) -> pd.Series:
    """Wilder 平滑（RSI/ATR 中使用）。
    
    Args:
        series: 输入序列
        period: 平滑周期
        
    Returns:
        平滑后序列
    """
    alpha = 1.0 / period
    result = np.zeros(len(series))
    result[:] = np.nan
    # 首个有效值使用简单平均初始化
    start_idx = period
    if start_idx < len(series):
        result[start_idx - 1] = series.iloc[:period].mean()
        for i in range(start_idx, len(series)):
            val = series.iloc[i]
            if np.isfinite(val):
                result[i] = alpha * val + (1 - alpha) * result[i - 1]
            else:
                result[i] = result[i - 1]
    return pd.Series(result, index=series.index)


# ==============================================================================
# 趋势指标 (Trend Indicators) ~25个
# ==============================================================================

def compute_ma(df: pd.DataFrame, periods: list = None) -> pd.DataFrame:
    """计算移动平均线 (SMA)。
    
    Args:
        df: OHLCV DataFrame
        periods: MA 周期列表，默认 [5, 10, 20, 60, 120]
        
    Returns:
        DataFrame with MA columns
    """
    if periods is None:
        periods = [5, 10, 20, 60, 120]
    result = pd.DataFrame(index=df.index)
    close = df["close"]
    for p in periods:
        result[f"MA{p}"] = _sma(close, p)
    return result


def compute_ema(df: pd.DataFrame) -> pd.DataFrame:
    """计算指数移动平均线 EMA12 和 EMA26。
    
    Args:
        df: OHLCV DataFrame
        
    Returns:
        DataFrame with EMA12, EMA26 columns
    """
    result = pd.DataFrame(index=df.index)
    close = df["close"]
    result["EMA12"] = _ema(close, 12)
    result["EMA26"] = _ema(close, 26)
    return result


def compute_macd(df: pd.DataFrame) -> pd.DataFrame:
    """计算 MACD 指标。
    
    MACD 线 (DIF) = EMA12 - EMA26
    信号线 (DEA) = DIF 的 9日 EMA
    柱状图 (MACD) = DIF - DEA
    
    Args:
        df: OHLCV DataFrame
        
    Returns:
        DataFrame with DIF, DEA, MACD columns
    """
    result = pd.DataFrame(index=df.index)
    close = df["close"]
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    dif = ema12 - ema26
    dea = _ema(dif, 9)
    result["DIF"] = dif
    result["DEA"] = dea
    result["MACD"] = 2.0 * (dif - dea)
    return result


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """计算 ADX (Average Directional Index) 及 +DI, -DI。
    
    Args:
        df: OHLCV DataFrame
        period: ADX 周期，默认 14
        
    Returns:
        DataFrame with ADX, PDI, MDI columns
    """
    result = pd.DataFrame(index=df.index)
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    n = len(close)

    # True Range
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    # Directional Movement
    up_move = np.zeros(n)
    down_move = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        if up > down and up > 0:
            up_move[i] = up
        if down > up and down > 0:
            down_move[i] = down

    # Wilder smoothed TR, +DM, -DM
    tr_smoothed = _wilder_smoothing(pd.Series(tr, index=df.index), period)
    up_smoothed = _wilder_smoothing(pd.Series(up_move, index=df.index), period)
    down_smoothed = _wilder_smoothing(pd.Series(down_move, index=df.index), period)

    # Directional Indicators
    pdi = _safe_divide(up_smoothed.values, tr_smoothed.values) * 100
    mdi = _safe_divide(down_smoothed.values, tr_smoothed.values) * 100

    # DX = |PDI - MDI| / (PDI + MDI) * 100
    dx = _safe_divide(np.abs(pdi - mdi), (pdi + mdi)) * 100

    # ADX = Wilder smoothed DX
    adx = _wilder_smoothing(pd.Series(dx, index=df.index), period)

    result["ADX"] = adx.values
    result["PDI"] = pdi
    result["MDI"] = mdi
    return result


def compute_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """计算 CCI (Commodity Channel Index)。
    
    CCI = (TP - SMA(TP)) / (0.015 * Mean Deviation)
    TP = (High + Low + Close) / 3
    
    Args:
        df: OHLCV DataFrame
        period: CCI 周期，默认 20
        
    Returns:
        DataFrame with CCI column
    """
    result = pd.DataFrame(index=df.index)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_sma = _sma(tp, period)
    mad = tp.rolling(window=period, min_periods=period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    result["CCI"] = _safe_divide(tp.values - tp_sma.values, 0.015 * mad.values)
    return result


def compute_roc(df: pd.DataFrame, period: int = 12) -> pd.DataFrame:
    """计算 ROC (Rate of Change)。
    
    ROC = (Close - Close_n) / Close_n * 100
    
    Args:
        df: OHLCV DataFrame
        period: ROC 周期，默认 12
        
    Returns:
        DataFrame with ROC column
    """
    result = pd.DataFrame(index=df.index)
    close = df["close"]
    result["ROC"] = (close - close.shift(period)) / close.shift(period) * 100
    return result


def compute_trix(df: pd.DataFrame, period: int = 15) -> pd.DataFrame:
    """计算 TRIX (Triple Exponential Average)。
    
    TRIX = ROC of triple-smoothed EMA
    
    Args:
        df: OHLCV DataFrame
        period: 周期，默认 15
        
    Returns:
        DataFrame with TRIX column
    """
    result = pd.DataFrame(index=df.index)
    close = df["close"]
    ema1 = _ema(close, period)
    ema2 = _ema(ema1, period)
    ema3 = _ema(ema2, period)
    result["TRIX"] = ema3.pct_change() * 100
    return result


def compute_dpo(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """计算 DPO (Detrended Price Oscillator)。
    
    DPO = Close - SMA(Close, period/2 + 1) shifted back
    
    Args:
        df: OHLCV DataFrame
        period: 周期，默认 20
        
    Returns:
        DataFrame with DPO column
    """
    result = pd.DataFrame(index=df.index)
    close = df["close"]
    shifted_period = period // 2 + 1
    sma = _sma(close, period)
    result["DPO"] = close - sma.shift(shifted_period)
    return result


def compute_kama(df: pd.DataFrame, period: int = 30) -> pd.DataFrame:
    """计算 KAMA (Kaufman's Adaptive Moving Average)。
    
    Args:
        df: OHLCV DataFrame
        period: 周期，默认 30
        
    Returns:
        DataFrame with KAMA column
    """
    result = pd.DataFrame(index=df.index)
    close = df["close"].values
    n = len(close)

    # Efficiency Ratio
    change = np.abs(close - np.roll(close, period))
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i - period:i + 1])))

    er = _safe_divide(change, volatility)

    # Smoothing Constant
    fast_sc = 2.0 / (2 + 1)   # fastest SC for period=2
    slow_sc = 2.0 / (30 + 1)  # slowest SC for period=30
    sc_raw = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    sc = np.where(np.isfinite(sc_raw), sc_raw, slow_sc)

    # KAMA
    kama = np.zeros(n)
    kama[:] = np.nan
    kama[period - 1] = close[period - 1]
    for i in range(period, n):
        if np.isfinite(sc[i]) and np.isfinite(close[i]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
        else:
            kama[i] = kama[i - 1]

    result["KAMA"] = kama
    return result


def compute_aroon(df: pd.DataFrame, period: int = 25) -> pd.DataFrame:
    """计算 Aroon 指标。
    
    Aroon Up = (period - days since highest high) / period * 100
    Aroon Down = (period - days since lowest low) / period * 100
    
    Args:
        df: OHLCV DataFrame
        period: 周期，默认 25
        
    Returns:
        DataFrame with AROON_UP, AROON_DOWN columns
    """
    result = pd.DataFrame(index=df.index)
    high = df["high"]
    low = df["low"]

    aroon_up = high.rolling(window=period + 1, min_periods=period + 1).apply(
        lambda x: (period - (len(x) - 1 - np.argmax(x))) / period * 100, raw=True
    )
    aroon_down = low.rolling(window=period + 1, min_periods=period + 1).apply(
        lambda x: (period - (len(x) - 1 - np.argmin(x))) / period * 100, raw=True
    )

    result["AROON_UP"] = aroon_up
    result["AROON_DOWN"] = aroon_down
    return result


def compute_bollinger(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """计算布林带 (Bollinger Bands)。
    
    Args:
        df: OHLCV DataFrame
        period: SMA 周期，默认 20
        num_std: 标准差倍数，默认 2
        
    Returns:
        DataFrame with BB_UPPER, BB_MID, BB_LOWER, BB_WIDTH, BB_PCT columns
    """
    result = pd.DataFrame(index=df.index)
    close = df["close"]
    mid = _sma(close, period)
    std = close.rolling(window=period, min_periods=period).std()

    upper = mid + num_std * std
    lower = mid - num_std * std

    result["BB_UPPER"] = upper
    result["BB_MID"] = mid
    result["BB_LOWER"] = lower
    result["BB_WIDTH"] = _safe_divide(upper.values - lower.values, mid.values)
    result["BB_PCT"] = _safe_divide(close.values - lower.values, upper.values - lower.values)
    return result


# ==============================================================================
# 动量指标 (Momentum Indicators) ~15个
# ==============================================================================

def compute_rsi(df: pd.DataFrame, periods: list = None) -> pd.DataFrame:
    """计算 RSI (Relative Strength Index)。
    
    使用 Wilder 平滑方法。
    
    Args:
        df: OHLCV DataFrame
        periods: RSI 周期列表，默认 [6, 14, 24]
        
    Returns:
        DataFrame with RSI columns
    """
    if periods is None:
        periods = [6, 14, 24]
    result = pd.DataFrame(index=df.index)
    close = df["close"]
    delta = close.diff()

    for period in periods:
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = _wilder_smoothing(gain, period)
        avg_loss = _wilder_smoothing(loss, period)
        rs = _safe_divide(avg_gain.values, avg_loss.values)
        rsi = 100.0 - 100.0 / (1.0 + rs)
        result[f"RSI{period}"] = rsi

    return result


def compute_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
    """计算 KDJ 随机指标。
    
    RSV = (Close - Low_n) / (High_n - Low_n) * 100
    K = EMA(RSV, m1)
    D = EMA(K, m2)
    J = 3*K - 2*D
    
    Args:
        df: OHLCV DataFrame
        n: RSV 周期，默认 9
        m1: K 平滑周期，默认 3
        m2: D 平滑周期，默认 3
        
    Returns:
        DataFrame with KDJ_K, KDJ_D, KDJ_J columns
    """
    result = pd.DataFrame(index=df.index)
    high_n = df["high"].rolling(window=n, min_periods=n).max()
    low_n = df["low"].rolling(window=n, min_periods=n).min()
    close = df["close"]

    rsv = _safe_divide(close.values - low_n.values, high_n.values - low_n.values) * 100
    rsv_series = pd.Series(rsv, index=df.index)

    k = rsv_series.ewm(alpha=1.0 / m1, adjust=False).mean()
    d = k.ewm(alpha=1.0 / m2, adjust=False).mean()
    j = 3.0 * k - 2.0 * d

    result["KDJ_K"] = k
    result["KDJ_D"] = d
    result["KDJ_J"] = j
    return result


def compute_wr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """计算 Williams %R。
    
    WR = (Highest High - Close) / (Highest High - Lowest Low) * -100
    
    Args:
        df: OHLCV DataFrame
        period: 周期，默认 14
        
    Returns:
        DataFrame with WR column
    """
    result = pd.DataFrame(index=df.index)
    high_n = df["high"].rolling(window=period, min_periods=period).max()
    low_n = df["low"].rolling(window=period, min_periods=period).min()
    close = df["close"]
    wr = _safe_divide(high_n.values - close.values, high_n.values - low_n.values) * -100
    result["WR"] = wr
    return result


def compute_mom(df: pd.DataFrame, periods: list = None) -> pd.DataFrame:
    """计算 MOM (Momentum) 价格动量。
    
    MOM = Close - Close_n
    
    Args:
        df: OHLCV DataFrame
        periods: 周期列表，默认 [10, 20]
        
    Returns:
        DataFrame with MOM columns
    """
    if periods is None:
        periods = [10, 20]
    result = pd.DataFrame(index=df.index)
    close = df["close"]
    for p in periods:
        result[f"MOM{p}"] = close - close.shift(p)
    return result


def compute_returns(df: pd.DataFrame, periods: list = None) -> pd.DataFrame:
    """计算多周期收益率。
    
    Args:
        df: OHLCV DataFrame
        periods: 周期列表，默认 [1, 5, 10, 20, 60]
        
    Returns:
        DataFrame with Ret columns
    """
    if periods is None:
        periods = [1, 5, 10, 20, 60]
    result = pd.DataFrame(index=df.index)
    close = df["close"]
    for p in periods:
        result[f"Ret{p}d"] = close.pct_change(periods=p) * 100
    return result


def compute_max_ret(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """计算滚动窗口内最大单日收益率。
    
    Args:
        df: OHLCV DataFrame
        period: 滚动窗口，默认 20
        
    Returns:
        DataFrame with MaxRet20d column
    """
    result = pd.DataFrame(index=df.index)
    daily_ret = df["close"].pct_change()
    result[f"MaxRet{period}d"] = daily_ret.rolling(
        window=period, min_periods=period
    ).max() * 100
    return result


# ==============================================================================
# 波动率指标 (Volatility Indicators) ~10个
# ==============================================================================

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """计算 ATR (Average True Range)。
    
    使用 Wilder 平滑方法。
    
    Args:
        df: OHLCV DataFrame
        period: 周期，默认 14
        
    Returns:
        DataFrame with ATR column
    """
    result = pd.DataFrame(index=df.index)
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    n = len(close)

    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    tr[0] = high[0] - low[0]

    tr_series = pd.Series(tr, index=df.index)
    atr = _wilder_smoothing(tr_series, period)
    result["ATR"] = atr.values
    return result


def compute_std(df: pd.DataFrame, periods: list = None) -> pd.DataFrame:
    """计算价格标准差。
    
    Args:
        df: OHLCV DataFrame
        periods: 周期列表，默认 [20, 60]
        
    Returns:
        DataFrame with STD columns
    """
    if periods is None:
        periods = [20, 60]
    result = pd.DataFrame(index=df.index)
    close = df["close"]
    for p in periods:
        result[f"STD{p}"] = close.rolling(window=p, min_periods=p).std()
    return result


def compute_historical_volatility(df: pd.DataFrame, periods: list = None,
                                   trading_days: int = 252) -> pd.DataFrame:
    """计算历史波动率 (年化)。
    
    HV = std(daily_returns) * sqrt(trading_days)
    
    Args:
        df: OHLCV DataFrame
        periods: 周期列表，默认 [20, 60]
        trading_days: 年化交易日数，默认 252
        
    Returns:
        DataFrame with HV columns
    """
    if periods is None:
        periods = [20, 60]
    result = pd.DataFrame(index=df.index)
    daily_ret = df["close"].pct_change()
    for p in periods:
        result[f"HV{p}"] = daily_ret.rolling(
            window=p, min_periods=p
        ).std() * np.sqrt(trading_days)
    return result


def compute_ulcer_index(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """计算 Ulcer Index。
    
    UI = sqrt(mean of squared percentage drawdowns)
    
    Args:
        df: OHLCV DataFrame
        period: 周期，默认 14
        
    Returns:
        DataFrame with UlcerIndex column
    """
    result = pd.DataFrame(index=df.index)
    close = df["close"]

    # 滚动窗口内的最高价
    rolling_max = close.rolling(window=period, min_periods=period).max()

    # 百分比回撤
    drawdown_pct = (close - rolling_max) / rolling_max * 100

    # Ulcer Index = sqrt(mean(drawdown^2))
    result["UlcerIndex"] = np.sqrt(
        (drawdown_pct ** 2).rolling(window=period, min_periods=period).mean()
    )
    return result


def compute_high_low_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """计算日内高低价比率。
    
    HighLowRatio = High / Low
    
    Args:
        df: OHLCV DataFrame
        
    Returns:
        DataFrame with HighLowRatio column
    """
    result = pd.DataFrame(index=df.index)
    result["HighLowRatio"] = _safe_divide(df["high"].values, df["low"].values)
    return result


# ==============================================================================
# 成交量/价格指标 (Volume/Price Indicators) ~15个
# ==============================================================================

def compute_obv(df: pd.DataFrame) -> pd.DataFrame:
    """计算 OBV (On-Balance Volume)。
    
    Args:
        df: OHLCV DataFrame
        
    Returns:
        DataFrame with OBV column
    """
    result = pd.DataFrame(index=df.index)
    close = df["close"]
    volume = df["volume"]

    direction = np.sign(close.diff().fillna(0))
    obv = (direction * volume).cumsum()
    result["OBV"] = obv
    return result


def compute_cmf(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """计算 CMF (Chaikin Money Flow)。
    
    MFM = ((Close - Low) - (High - Close)) / (High - Low)
    MFV = MFM * Volume
    CMF = SUM(MFV, period) / SUM(Volume, period)
    
    Args:
        df: OHLCV DataFrame
        period: 周期，默认 20
        
    Returns:
        DataFrame with CMF column
    """
    result = pd.DataFrame(index=df.index)
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    volume = df["volume"].values

    hl_range = high - low
    mfm = _safe_divide((close - low) - (high - close), hl_range)
    mfv = mfm * volume

    mfv_series = pd.Series(mfv, index=df.index)
    volume_series = pd.Series(volume, index=df.index)

    result["CMF"] = _safe_divide(
        mfv_series.rolling(window=period, min_periods=period).sum().values,
        volume_series.rolling(window=period, min_periods=period).sum().values,
    )
    return result


def compute_mfi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """计算 MFI (Money Flow Index)。
    
    TP = (High + Low + Close) / 3
    MF = TP * Volume
    MFI = 100 - 100 / (1 + Positive MF / Negative MF)
    
    Args:
        df: OHLCV DataFrame
        period: 周期，默认 14
        
    Returns:
        DataFrame with MFI column
    """
    result = pd.DataFrame(index=df.index)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    mf = tp * df["volume"]

    tp_diff = tp.diff()
    pos_flow = mf.where(tp_diff > 0, 0.0)
    neg_flow = mf.where(tp_diff < 0, 0.0)

    pos_sum = pos_flow.rolling(window=period, min_periods=period).sum()
    neg_sum = neg_flow.rolling(window=period, min_periods=period).sum()

    mr = _safe_divide(pos_sum.values, neg_sum.values)
    result["MFI"] = 100.0 - 100.0 / (1.0 + mr)
    return result


def compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """计算 VWAP (Volume Weighted Average Price)。
    
    VWAP = cumulative(TP * Volume) / cumulative(Volume)
    
    Args:
        df: OHLCV DataFrame
        
    Returns:
        DataFrame with VWAP column
    """
    result = pd.DataFrame(index=df.index)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_pv = (tp * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    result["VWAP"] = _safe_divide(cum_pv.values, cum_vol.values)
    return result


def compute_vroc(df: pd.DataFrame, period: int = 12) -> pd.DataFrame:
    """计算 VROC (Volume Rate of Change)。
    
    Args:
        df: OHLCV DataFrame
        period: 周期，默认 12
        
    Returns:
        DataFrame with VROC column
    """
    result = pd.DataFrame(index=df.index)
    volume = df["volume"]
    result["VROC"] = (volume - volume.shift(period)) / volume.shift(period) * 100
    return result


def compute_turnover(df: pd.DataFrame, periods: list = None) -> pd.DataFrame:
    """计算换手率相关指标。
    
    如果 DataFrame 中有 'turnover_rate' 列则直接使用，否则使用 volume 作为代理。
    
    Args:
        df: OHLCV DataFrame
        periods: 周期列表，默认 [5, 20]
        
    Returns:
        DataFrame with Turnover columns
    """
    if periods is None:
        periods = [5, 20]
    result = pd.DataFrame(index=df.index)

    if "turnover_rate" in df.columns:
        turnover = df["turnover_rate"]
    else:
        # 使用成交量作为换手率代理（标准化）
        turnover = df["volume"] / df["volume"].rolling(window=252, min_periods=60).mean()

    for p in periods:
        result[f"Turnover{p}"] = turnover.rolling(window=p, min_periods=1).mean()
    return result


def compute_amount_ma(df: pd.DataFrame, periods: list = None) -> pd.DataFrame:
    """计算成交额移动平均。
    
    Args:
        df: OHLCV DataFrame
        periods: 周期列表，默认 [5, 20]
        
    Returns:
        DataFrame with AmountMA columns
    """
    if periods is None:
        periods = [5, 20]
    result = pd.DataFrame(index=df.index)
    if "amount" not in df.columns:
        for p in periods:
            result[f"AmountMA{p}"] = np.nan
        return result
    amount = df["amount"]
    for p in periods:
        result[f"AmountMA{p}"] = _sma(amount, p)
    return result


def compute_volume_ratio(df: pd.DataFrame, periods: list = None) -> pd.DataFrame:
    """计算成交量比率 (volume / n-day average volume)。
    
    Args:
        df: OHLCV DataFrame
        periods: 周期列表，默认 [5, 20]
        
    Returns:
        DataFrame with VolumeRatio columns
    """
    if periods is None:
        periods = [5, 20]
    result = pd.DataFrame(index=df.index)
    volume = df["volume"]
    for p in periods:
        avg_vol = _sma(volume, p)
        result[f"VolumeRatio{p}"] = _safe_divide(volume.values, avg_vol.values)
    return result


# ==============================================================================
# 统计指标 (Statistical Indicators) ~12个
# ==============================================================================

def compute_skewness(df: pd.DataFrame, periods: list = None) -> pd.DataFrame:
    """计算收益率的偏度。
    
    Args:
        df: OHLCV DataFrame
        periods: 周期列表，默认 [20, 60]
        
    Returns:
        DataFrame with Skew columns
    """
    if periods is None:
        periods = [20, 60]
    result = pd.DataFrame(index=df.index)
    daily_ret = df["close"].pct_change()
    for p in periods:
        result[f"Skew{p}"] = daily_ret.rolling(
            window=p, min_periods=p // 2
        ).skew()
    return result


def compute_kurtosis(df: pd.DataFrame, periods: list = None) -> pd.DataFrame:
    """计算收益率的峰度。
    
    Args:
        df: OHLCV DataFrame
        periods: 周期列表，默认 [20, 60]
        
    Returns:
        DataFrame with Kurt columns
    """
    if periods is None:
        periods = [20, 60]
    result = pd.DataFrame(index=df.index)
    daily_ret = df["close"].pct_change()
    for p in periods:
        result[f"Kurt{p}"] = daily_ret.rolling(
            window=p, min_periods=p // 2
        ).kurt()
    return result


def compute_beta_alpha(df: pd.DataFrame, periods: list = None,
                        market_returns: pd.Series = None) -> pd.DataFrame:
    """计算 Beta 和 Alpha (Jensen's Alpha)。
    
    使用等权组合作为市场代理（如果没有提供市场收益率）。
    
    Args:
        df: OHLCV DataFrame
        periods: 周期列表，默认 [60, 120]
        market_returns: 市场收益率序列。如果为 None，使用等权组合代理
        
    Returns:
        DataFrame with Beta, Alpha columns
    """
    if periods is None:
        periods = [60, 120]
    result = pd.DataFrame(index=df.index)
    daily_ret = df["close"].pct_change()

    # 如果没有市场收益率，使用等权代理（自身收益率作为代理）
    if market_returns is None:
        market_returns = daily_ret.copy()

    for p in periods:
        # 计算滚动 Beta = Cov(ret, market) / Var(market)
        cov = daily_ret.rolling(window=p, min_periods=p // 2).cov(market_returns)
        market_var = market_returns.rolling(window=p, min_periods=p // 2).var()
        beta = _safe_divide(cov.values, market_var.values)
        result[f"Beta{p}"] = beta

        # Alpha = mean(ret) - beta * mean(market)
        ret_mean = daily_ret.rolling(window=p, min_periods=p // 2).mean()
        market_mean = market_returns.rolling(window=p, min_periods=p // 2).mean()
        if isinstance(ret_mean, pd.Series):
            alpha = ret_mean.values - beta * market_mean.values
        else:
            alpha = ret_mean - beta * market_mean
        result[f"Alpha{p}"] = alpha

    return result


def compute_sharpe(df: pd.DataFrame, periods: list = None,
                    rf_rate: float = 0.0, trading_days: int = 252) -> pd.DataFrame:
    """计算滚动 Sharpe Ratio。
    
    Sharpe = (mean(ret - rf) / std(ret)) * sqrt(trading_days)
    
    Args:
        df: OHLCV DataFrame
        periods: 周期列表，默认 [20, 60]
        rf_rate: 日化无风险利率，默认 0
        trading_days: 年化交易日数，默认 252
        
    Returns:
        DataFrame with Sharpe columns
    """
    if periods is None:
        periods = [20, 60]
    result = pd.DataFrame(index=df.index)
    daily_ret = df["close"].pct_change()
    excess = daily_ret - rf_rate
    for p in periods:
        ret_mean = excess.rolling(window=p, min_periods=p).mean()
        ret_std = excess.rolling(window=p, min_periods=p).std()
        result[f"Sharpe{p}"] = _safe_divide(
            ret_mean.values, ret_std.values
        ) * np.sqrt(trading_days)
    return result


def compute_correlation(df: pd.DataFrame, period: int = 20,
                         market_returns: pd.Series = None) -> pd.DataFrame:
    """计算与市场（等权组合）的滚动相关性。
    
    Args:
        df: OHLCV DataFrame
        period: 滚动窗口，默认 20
        market_returns: 市场收益率。如果为 None，使用自身收益率
        
    Returns:
        DataFrame with Corr column
    """
    result = pd.DataFrame(index=df.index)
    daily_ret = df["close"].pct_change()
    if market_returns is None:
        market_returns = daily_ret.copy()
    result[f"Corr{period}"] = daily_ret.rolling(
        window=period, min_periods=period // 2
    ).corr(market_returns)
    return result


def compute_max_drawdown(df: pd.DataFrame, period: int = 60) -> pd.DataFrame:
    """计算滚动最大回撤。
    
    Args:
        df: OHLCV DataFrame
        period: 滚动窗口，默认 60
        
    Returns:
        DataFrame with MaxDD60 column
    """
    result = pd.DataFrame(index=df.index)
    close = df["close"]

    def _max_dd(x):
        cummax = np.maximum.accumulate(x)
        drawdowns = (cummax - x) / cummax
        return np.max(drawdowns) if len(drawdowns) > 0 else np.nan

    result[f"MaxDD{period}"] = close.rolling(
        window=period, min_periods=period // 2
    ).apply(_max_dd, raw=True)
    return result


def compute_calmar_ratio(df: pd.DataFrame, period: int = 60,
                          trading_days: int = 252) -> pd.DataFrame:
    """计算 Calmar Ratio (年化收益率 / 最大回撤)。
    
    Args:
        df: OHLCV DataFrame
        period: 滚动窗口，默认 60
        trading_days: 年化交易日数，默认 252
        
    Returns:
        DataFrame with CalmarRatio column
    """
    result = pd.DataFrame(index=df.index)
    close = df["close"]
    daily_ret = close.pct_change()

    annual_return = daily_ret.rolling(
        window=period, min_periods=period // 2
    ).mean() * trading_days

    max_dd_df = compute_max_drawdown(df, period)
    max_dd = max_dd_df[f"MaxDD{period}"]

    result[f"CalmarRatio{period}"] = _safe_divide(annual_return.values, max_dd.values)
    return result


# ==============================================================================
# 主计算器类
# ==============================================================================

class TechnicalFactorCalculator:
    """技术因子计算器。

    给定包含 [open, high, low, close, volume, amount] 列的 DataFrame，
    计算全部 100+ 技术因子。

    因子分类：
        - 趋势指标 (Trend): MA, EMA, MACD, ADX, CCI, ROC, TRIX, DPO, KAMA, Aroon, BB
        - 动量指标 (Momentum): RSI, KDJ, WR, MOM, Returns, MaxRet
        - 波动率指标 (Volatility): ATR, STD, HV, UlcerIndex, HighLowRatio
        - 成交量/价格 (Volume/Price): OBV, CMF, MFI, VWAP, VROC, Turnover, AmountMA, VolumeRatio
        - 统计指标 (Statistical): Skew, Kurt, Beta, Alpha, Sharpe, Corr, MaxDD, CalmarRatio

    Usage:
        >>> calc = TechnicalFactorCalculator()
        >>> factors = calc.compute_all(df)
    """

    def __init__(self, market_returns: pd.Series = None):
        """初始化计算器。

        Args:
            market_returns: 市场收益率序列（用于 Beta/Alpha/Corr 计算）。
                           如果为 None，使用等权组合代理。
        """
        self.market_returns = market_returns
        self._factor_list = []

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算全部技术因子。

        Args:
            df: OHLCV DataFrame，必须包含 [open, high, low, close, volume] 列，
                amount 列可选

        Returns:
            DataFrame，每一列是一个因子，按日期索引

        Raises:
            ValueError: 如果数据为空或缺少必要列
        """
        if df.empty:
            raise ValueError("输入 DataFrame 为空")

        required_cols = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"缺少必要列: {missing}")

        logger.info("开始计算技术因子，数据形状: %s", df.shape)

        # 排序并去重
        df = df.sort_index().copy()
        df = df[~df.index.duplicated(keep="first")]

        # 处理无穷值
        df = df.replace([np.inf, -np.inf], np.nan)

        # 按顺序计算各类因子
        trend = self.compute_trend(df)
        momentum = self.compute_momentum(df)
        volatility = self.compute_volatility(df)
        volume_price = self.compute_volume_price(df)
        statistical = self.compute_statistical(df)

        # 合并所有因子
        all_factors = pd.concat(
            [trend, momentum, volatility, volume_price, statistical],
            axis=1,
        )

        # 清理异常值
        all_factors = self._clean_factors(all_factors)

        self._factor_list = list(all_factors.columns)
        logger.info("技术因子计算完成，共 %d 个因子", len(self._factor_list))

        return all_factors

    def _clean_factors(self, factors: pd.DataFrame) -> pd.DataFrame:
        """清理因子数据：替换无穷值、极端值裁剪。

        Args:
            factors: 因子 DataFrame

        Returns:
            清理后的 DataFrame
        """
        factors = factors.replace([np.inf, -np.inf], np.nan)
        # 对极端值进行 winsorize 裁剪 (1% / 99%)
        for col in factors.columns:
            series = factors[col].copy()
            mask = series.notna()
            if mask.sum() <= 10:
                continue
            valid = series[mask]
            lower = valid.quantile(0.01)
            upper = valid.quantile(0.99)
            series[series < lower] = lower
            series[series > upper] = upper
            factors[col] = series
        return factors

    def compute_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算趋势类指标。

        包括: MA(5/10/20/60/120), EMA(12/26), MACD, ADX, CCI,
              ROC, TRIX, DPO, KAMA, Aroon, Bollinger Bands

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with trend factors
        """
        logger.info("计算趋势指标...")
        factors = []

        factors.append(compute_ma(df))
        factors.append(compute_ema(df))
        factors.append(compute_macd(df))
        factors.append(compute_adx(df))
        factors.append(compute_cci(df))
        factors.append(compute_roc(df))
        factors.append(compute_trix(df))
        factors.append(compute_dpo(df))
        factors.append(compute_kama(df))
        factors.append(compute_aroon(df))
        factors.append(compute_bollinger(df))

        return pd.concat(factors, axis=1)

    def compute_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算动量类指标。

        包括: RSI(6/14/24), KDJ, WR, MOM(10/20),
              Ret(1/5/10/20/60)d, MaxRet20d

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with momentum factors
        """
        logger.info("计算动量指标...")
        factors = []

        factors.append(compute_rsi(df))
        factors.append(compute_kdj(df))
        factors.append(compute_wr(df))
        factors.append(compute_mom(df))
        factors.append(compute_returns(df))
        factors.append(compute_max_ret(df))

        return pd.concat(factors, axis=1)

    def compute_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算波动率类指标。

        包括: ATR, STD(20/60), HV(20/60), UlcerIndex, HighLowRatio

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with volatility factors
        """
        logger.info("计算波动率指标...")
        factors = []

        factors.append(compute_atr(df))
        factors.append(compute_std(df))
        factors.append(compute_historical_volatility(df))
        factors.append(compute_ulcer_index(df))
        factors.append(compute_high_low_ratio(df))

        return pd.concat(factors, axis=1)

    def compute_volume_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算成交量/价格类指标。

        包括: OBV, CMF, MFI, VWAP, VROC, Turnover(5/20),
              AmountMA(5/20), VolumeRatio(5/20)

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with volume/price factors
        """
        logger.info("计算成交量/价格指标...")
        factors = []

        factors.append(compute_obv(df))
        factors.append(compute_cmf(df))
        factors.append(compute_mfi(df))
        factors.append(compute_vwap(df))
        factors.append(compute_vroc(df))
        factors.append(compute_turnover(df))
        factors.append(compute_amount_ma(df))
        factors.append(compute_volume_ratio(df))

        return pd.concat(factors, axis=1)

    def compute_statistical(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算统计类指标。

        包括: Skew(20/60), Kurt(20/60), Beta(60/120), Alpha(60),
              Sharpe(20/60), Corr20, MaxDD60, CalmarRatio60

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with statistical factors
        """
        logger.info("计算统计指标...")
        factors = []

        factors.append(compute_skewness(df))
        factors.append(compute_kurtosis(df))
        factors.append(compute_beta_alpha(df, market_returns=self.market_returns))
        factors.append(compute_sharpe(df))
        factors.append(compute_correlation(df, market_returns=self.market_returns))
        factors.append(compute_max_drawdown(df))
        factors.append(compute_calmar_ratio(df))

        return pd.concat(factors, axis=1)

    @property
    def factor_names(self) -> list:
        """返回已计算的所有因子名称列表。"""
        return self._factor_list

    @property
    def factor_count(self) -> int:
        """返回因子数量。"""
        return len(self._factor_list)


# ==============================================================================
# 便捷入口
# ==============================================================================

def compute_all_factors(df: pd.DataFrame,
                         market_returns: pd.Series = None) -> pd.DataFrame:
    """便捷函数：一次性计算所有技术因子。

    Args:
        df: OHLCV DataFrame
        market_returns: 市场收益率序列（可选）

    Returns:
        DataFrame with all factors
    """
    calc = TechnicalFactorCalculator(market_returns=market_returns)
    return calc.compute_all(df)