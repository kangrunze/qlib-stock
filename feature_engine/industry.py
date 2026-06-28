# -*- coding: utf-8 -*-
"""
行业因子 — IndustryFactorCalculator

计算行业分类、行业中性化、行业轮动等因子。

状态: 占位模块，待行业数据接入后实现
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class IndustryCalculator:
    """行业因子计算器（占位）"""

    def __init__(self):
        self.is_fitted = False

    def compute(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        logger.warning("IndustryCalculator: 数据源未接入，返回空 DataFrame")
        return None

    def fit(self, *args, **kwargs):
        self.is_fitted = True
        return self