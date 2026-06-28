# -*- coding: utf-8 -*-
"""
Feature Engine - 自定义因子层

本模块仅保留 Alpha158/Alpha360 未覆盖的因子类别：
  - 资金流 (fund_flow)      — 北向资金、主力资金、融资融券
  - 财务 (fundamental)       — 估值、盈利、成长、质量
  - 行业 (industry)          — 行业分类、行业中性化
  - 新闻 (news)              — NLP 情绪得分
  - 公告 (announcements)     — 业绩预告、重大事项
  - 另类数据 (alternative)   — 舆情、供应链、ESG

技术指标、动量、波动率、量价、Alpha 公式等已由 Qlib Alpha158/Alpha360 覆盖，
不再重复实现。如需自定义技术因子，请通过 Qlib custom handler 扩展。

Usage:
    from feature_engine import FundamentalCalculator, FactorStore
"""

from .fundamental import FundamentalFactorCalculator
from .factor_store import FactorStore

__all__ = [
    "FundamentalFactorCalculator",
    "FactorStore",
]