# -*- coding: utf-8 -*-
"""
公告因子 — AnnouncementFactorCalculator

基于业绩预告、重大事项公告提取事件驱动因子。

状态: 占位模块，待公告数据接入后实现
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AnnouncementCalculator:
    """公告因子计算器（占位）"""

    def __init__(self):
        self.is_fitted = False

    def compute(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        logger.warning("AnnouncementCalculator: 数据源未接入，返回空 DataFrame")
        return None

    def fit(self, *args, **kwargs):
        self.is_fitted = True
        return self