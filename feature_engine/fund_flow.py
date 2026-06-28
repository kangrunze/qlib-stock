# -*- coding: utf-8 -*-
"""
资金流因子 — FundFlowFactorCalculator

覆盖 Qlib Alpha158 未包含的资金流信号：
  - 北向资金（沪股通/深股通）净流入
  - 主力资金净流入（超大单+大单）
  - 融资融券余额变化
  - 大宗交易折溢价

输入: 数据源 API（待实现）
输出: pd.DataFrame indexed by date with fund flow factors

状态: 占位模块，待数据源接入后实现
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FundFlowCalculator:
    """资金流因子计算器（占位）。

    预计实现因子:
      - net_inflow_5d / 20d      : 北向资金净流入（5日/20日累计）
      - main_force_ratio         : 主力资金净流入占比
      - margin_balance_change    : 融资余额变化率
      - block_trade_premium      : 大宗交易平均溢价率
    """

    def __init__(self):
        self.is_fitted = False

    def compute(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """计算资金流因子（当前为占位实现）。"""
        logger.warning("FundFlowCalculator: 数据源未接入，返回空 DataFrame")
        return None

    def fit(self, *args, **kwargs):
        self.is_fitted = True
        return self