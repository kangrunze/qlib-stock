# -*- coding: utf-8 -*-
"""
新闻情绪因子 — NewsSentimentCalculator

基于 NLP 的新闻情绪分析，提取情感得分、话题热度等因子。

状态: 占位模块，待 NLP 模型接入后实现
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class NewsSentimentCalculator:
    """新闻情绪因子计算器（占位）"""

    def __init__(self):
        self.is_fitted = False

    def compute(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        logger.warning("NewsSentimentCalculator: 数据源未接入，返回空 DataFrame")
        return None

    def fit(self, *args, **kwargs):
        self.is_fitted = True
        return self