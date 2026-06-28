# -*- coding: utf-8 -*-
"""
另类数据因子 — AlternativeDataCalculator

覆盖非传统数据源：
  - 舆情监测（微博/雪球/东方财富吧）
  - 供应链关系
  - ESG 评级
  - 专利与研发

状态: 占位模块，待数据源接入后实现
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AlternativeDataCalculator:
    """另类数据因子计算器（占位）"""

    def __init__(self):
        self.is_fitted = False

    def compute(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        logger.warning("AlternativeDataCalculator: 数据源未接入，返回空 DataFrame")
        return None

    def fit(self, *args, **kwargs):
        self.is_fitted = True
        return self