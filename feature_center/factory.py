# -*- coding: utf-8 -*-
"""因子工厂 - 动态生成因子"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureFactory:
    """因子工厂 - 根据配置动态生成因子"""
    
    def __init__(self, registry=None):
        self.registry = registry
        self.compute_cache = {}
        
    def compute_feature(
        self,
        feature_name: str,
        data: pd.DataFrame,
        params: Optional[Dict] = None
    ) -> pd.Series:
        """计算单个因子
        
        Args:
            feature_name: 因子名称
            data: 输入数据 (columns: open, high, low, close, volume)
            params: 参数覆盖
            
        Returns:
            因子值 Series
        """
        # 检查缓存
        cache_key = f"{feature_name}_{hash(str(params))}"
        if cache_key in self.compute_cache:
            return self.compute_cache[cache_key]
            
        # 获取参数
        if params is None and self.registry:
            feature_def = self.registry.get_feature(feature_name)
            if feature_def:
                params = feature_def.get("params", {})
                
        params = params or {}
        
        # 计算因子
        if feature_name.startswith("MA"):
            window = params.get("window", int(feature_name[2:]))
            result = self._compute_ma(data["close"], window)
        elif feature_name == "MACD":
            fast = params.get("fast", 12)
            slow = params.get("slow", 26)
            signal = params.get("signal", 9)
            result = self._compute_macd(data["close"], fast, slow, signal)
        elif feature_name == "RSI":
            window = params.get("window", 14)
            result = self._compute_rsi(data["close"], window)
        elif feature_name == "KDJ":
            n = params.get("n", 9)
            result = self._compute_kdj(data, n)
        elif feature_name == "ATR":
            window = params.get("window", 14)
            result = self._compute_atr(data, window)
        elif feature_name == "BOLL_WIDTH":
            window = params.get("window", 20)
            nbdev = params.get("nbdev", 2)
            result = self._compute_boll_width(data["close"], window, nbdev)
        elif feature_name == "OBV":
            result = self._compute_obv(data)
        elif feature_name == "MFI":
            window = params.get("window", 14)
            result = self._compute_mfi(data, window)
        elif feature_name == "SKEW":
            window = params.get("window", 20)
            result = data["close"].rolling(window).skew()
        elif feature_name == "KURT":
            window = params.get("window", 20)
            result = data["close"].rolling(window).kurt()
        elif feature_name == "HIST_VOL":
            window = params.get("window", 20)
            returns = data["close"].pct_change()
            result = returns.rolling(window).std() * np.sqrt(252)
        else:
            logger.warning(f"未知的因子: {feature_name}")
            return pd.Series(np.nan, index=data.index)
            
        # 缓存结果
        self.compute_cache[cache_key] = result
        
        return result
        
    def _compute_ma(self, close: pd.Series, window: int) -> pd.Series:
        """计算移动平均"""
        return close.rolling(window).mean()
        
    def _compute_macd(
        self,
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.Series:
        """计算MACD"""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        hist = macd - signal_line
        return hist
        
    def _compute_rsi(self, close: pd.Series, window: int = 14) -> pd.Series:
        """计算RSI"""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def _compute_kdj(self, data: pd.DataFrame, n: int = 9) -> pd.Series:
        """计算KDJ的K值"""
        low_min = data["low"].rolling(n).min()
        high_max = data["high"].rolling(n).max()
        rsv = (data["close"] - low_min) / (high_max - low_min) * 100
        k = rsv.ewm(com=2, adjust=False).mean()
        return k
        
    def _compute_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """计算ATR"""
        high_low = data["high"] - data["low"]
        high_close = np.abs(data["high"] - data["close"].shift())
        low_close = np.abs(data["low"] - data["close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(window).mean()
        return atr
        
    def _compute_boll_width(
        self,
        close: pd.Series,
        window: int = 20,
        nbdev: int = 2
    ) -> pd.Series:
        """计算布林带宽度"""
        ma = close.rolling(window).mean()
        std = close.rolling(window).std()
        upper = ma + nbdev * std
        lower = ma - nbdev * std
        width = (upper - lower) / ma
        return width
        
    def _compute_obv(self, data: pd.DataFrame) -> pd.Series:
        """计算OBV"""
        direction = np.sign(data["close"].diff())
        obv = (direction * data["volume"]).cumsum()
        return obv
        
    def _compute_mfi(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """计算MFI"""
        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        money_flow = typical_price * data["volume"]
        
        positive_flow = money_flow.where(typical_price.diff() > 0, 0).rolling(window).sum()
        negative_flow = money_flow.where(typical_price.diff() < 0, 0).rolling(window).sum()
        
        money_ratio = positive_flow / negative_flow
        mfi = 100 - (100 / (1 + money_ratio))
        return mfi
        
    def compute_all_features(
        self,
        data: pd.DataFrame,
        feature_names: Optional[list] = None
    ) -> pd.DataFrame:
        """计算所有因子
        
        Args:
            data: 输入数据
            feature_names: 要计算的因子列表（默认全部）
            
        Returns:
            因子DataFrame
        """
        if feature_names is None and self.registry:
            features_df = self.registry.list_features()
            feature_names = features_df["name"].tolist()
            
        if feature_names is None:
            feature_names = ["MA5", "MA10", "MA20", "RSI", "MACD", "ATR"]
            
        result = pd.DataFrame(index=data.index)
        
        for feature_name in feature_names:
            try:
                result[feature_name] = self.compute_feature(feature_name, data)
            except Exception as e:
                logger.error(f"计算因子 {feature_name} 失败: {e}")
                result[feature_name] = np.nan
                
        return result
