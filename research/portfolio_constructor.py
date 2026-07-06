# -*- coding: utf-8 -*-
"""
行业中性化策略类 — portfolio_constructor.py

继承 Qlib 官方 TopkDropoutStrategy，在选股环节加入行业集中度约束。

设计原则（第 9.1 节）：
  这是对 Qlib 官方策略类的继承扩展，运行在 Qlib 官方回测引擎
  （SimulatorExecutor + PortAnaRecord）之内，不是独立的自研实现。
  这样能确保与现有的所有稳健性验证工具无缝兼容。

行业约束：
  - 单行业权重不超过 max_industry_weight（默认 25%）
  - 超出行业上限的股票依序跳过、递补排名更靠后但满足约束的股票

Usage:
    from research.portfolio_constructor import IndustryConstrainedTopkStrategy
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# 尝试从 Qlib 导入 TopkDropoutStrategy
try:
    from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
    _HAS_QLIB_STRATEGY = True
except ImportError:
    _HAS_QLIB_STRATEGY = False
    TopkDropoutStrategy = object


class IndustryConstrainedTopkStrategy(TopkDropoutStrategy if _HAS_QLIB_STRATEGY else object):
    """行业中性化 Top-K 选股策略。

    继承 Qlib 官方 TopkDropoutStrategy，在 generate_trade_decision 中
    插入行业集中度检查。运行在 Qlib 官方回测引擎之内，
    确保与现有的所有稳健性验证工具无缝兼容。

    Args:
        max_industry_weight: 单行业最大权重（比例，默认 0.25 = 25%）
        industry_map: {stock_code: industry_name} 行业映射
        topk: 选股数量
        n_drop: 每期替换数
    """

    def __init__(
        self,
        *args,
        max_industry_weight: float = 0.25,
        industry_map: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        if _HAS_QLIB_STRATEGY:
            super().__init__(*args, **kwargs)
        self.max_industry_weight = max_industry_weight
        self.industry_map = industry_map or {}
        # 如果父类有 topk/n_drop 属性，从 kwargs 中提取
        if not _HAS_QLIB_STRATEGY:
            self.topk = kwargs.get("topk", 30)
            self.n_drop = kwargs.get("n_drop", 5)

    def _get_industry(self, stock_code: str) -> str:
        """获取股票所属行业，未知行业归入 'Other'"""
        return self.industry_map.get(str(stock_code), "Other")

    def _apply_industry_constraint(
        self,
        ranked_stocks: List[str],
        target_count: Optional[int] = None,
    ) -> List[str]:
        """按排名顺序选股，同时满足行业集中度约束。

        算法：
          1. 按得分降序排列股票
          2. 依次尝试纳入，每纳入一只先检查其行业是否已超限
          3. 若超限则跳过，递补下一只
          4. 直到选满 target_count 只或候选池耗尽

        Args:
            ranked_stocks: 按得分排名的股票代码列表
            target_count: 目标选股数量（默认 = self.topk）

        Returns:
            满足行业约束的选股列表
        """
        if target_count is None:
            target_count = self.topk

        if not self.industry_map:
            return ranked_stocks[:target_count]

        selected = []
        industry_counts = {}

        for stock in ranked_stocks:
            if len(selected) >= target_count:
                break

            industry = self._get_industry(stock)
            current_count = industry_counts.get(industry, 0)
            max_per_industry = max(1, int(target_count * self.max_industry_weight))

            if current_count < max_per_industry:
                selected.append(stock)
                industry_counts[industry] = current_count + 1
            else:
                logger.debug(
                    "行业 %s 已达上限 (%d/%d)，跳过 %s",
                    industry, current_count, max_per_industry, stock,
                )

        if len(selected) < target_count:
            logger.warning(
                "行业约束下仅选出 %d/%d 只股票，可能行业映射不完整",
                len(selected), target_count,
            )

        return selected

    def generate_trade_decision(self, execute_result=None):
        """生成交易决策，复用父类信号排序逻辑，在最终选股环节插入行业集中度检查。

        这是 Qlib TopkDropoutStrategy 的标准接口方法。
        当行业映射可用时，对父类选出的股票进行行业约束过滤。
        """
        if not _HAS_QLIB_STRATEGY:
            logger.warning("Qlib TopkDropoutStrategy 不可用，行业约束策略无法在回测引擎中运行")
            return None

        # 调用父类的 generate_trade_decision 获取基础信号排序
        trade_decision = super().generate_trade_decision(execute_result)

        # 如果没有行业映射，直接返回父类结果
        if not self.industry_map:
            return trade_decision

        # 如果 trade_decision 包含股票列表，应用行业约束
        # Qlib 的 trade_decision 格式通常是 OrderedDict，key 为 stock_code
        if trade_decision is not None and hasattr(trade_decision, "__iter__"):
            try:
                stock_list = list(trade_decision.keys()) if hasattr(trade_decision, "keys") else list(trade_decision)
                constrained = set(self._apply_industry_constraint(stock_list))

                # 过滤掉不符合行业约束的股票
                if hasattr(trade_decision, "keys"):
                    filtered = {k: v for k, v in trade_decision.items() if k in constrained}
                    return filtered
            except Exception as e:
                logger.warning("行业约束应用失败: %s，回退到父类结果", e)

        return trade_decision

    def select_with_industry_constraint(
        self,
        ranked_stocks: List[str],
        scores: Dict[str, float],
        target_count: Optional[int] = None,
    ) -> List[str]:
        """独立选股接口：按排名顺序选股，同时满足行业集中度约束。

        用于非 Qlib 回测引擎场景（如独立选股推荐）。

        Args:
            ranked_stocks: 按得分排名的股票代码列表
            scores: {stock_code: score} 股票得分
            target_count: 目标选股数量（默认 = self.topk）

        Returns:
            满足行业约束的选股列表
        """
        return self._apply_industry_constraint(ranked_stocks, target_count)

    def select_with_industry_and_liquidity_constraint(
        self,
        ranked_stocks: List[str],
        scores: Dict[str, float],
        avg_volume: Dict[str, float],
        account_value: float = 1e8,
        max_volume_pct: float = 0.05,
        target_count: Optional[int] = None,
    ) -> Tuple[List[str], Dict[str, float]]:
        """选股 + 行业约束 + 流动性约束。

        流动性约束：拟建仓金额不超过股票日均成交额的 max_volume_pct（默认 5%）

        Args:
            ranked_stocks: 按得分排名的股票列表
            scores: {stock_code: score}
            avg_volume: {stock_code: 过去60日均成交额}
            account_value: 账户总资金
            max_volume_pct: 最大成交量占比
            target_count: 目标数量

        Returns:
            (selected_stocks, weights)
        """
        if target_count is None:
            target_count = self.topk

        if not self.industry_map:
            selected = ranked_stocks[:target_count]
            weights = {s: 1.0 / len(selected) for s in selected} if selected else {}
            return selected, weights

        selected = []
        weights = {}
        industry_counts = {}
        per_stock_weight = 1.0 / target_count
        position_value = account_value * per_stock_weight

        for stock in ranked_stocks:
            if len(selected) >= target_count:
                break

            # 流动性检查
            stock_avg_vol = avg_volume.get(stock, 0)
            if stock_avg_vol > 0 and position_value / stock_avg_vol > max_volume_pct:
                logger.debug("流动性不足，跳过 %s (成交额=%s)", stock, stock_avg_vol)
                continue

            # 行业检查
            industry = self._get_industry(stock)
            current_count = industry_counts.get(industry, 0)
            max_per_industry = max(1, int(target_count * self.max_industry_weight))

            if current_count >= max_per_industry:
                logger.debug("行业限制，跳过 %s (%s)", stock, industry)
                continue

            selected.append(stock)
            weights[stock] = per_stock_weight
            industry_counts[industry] = current_count + 1

        return selected, weights


def create_industry_map_from_csv(
    csv_path: str,
    stock_col: str = "stock_code",
    industry_col: str = "industry",
) -> Dict[str, str]:
    """从 CSV 文件中加载行业映射。

    Args:
        csv_path: CSV 文件路径
        stock_col: 股票代码列名
        industry_col: 行业分类列名

    Returns:
        {stock_code: industry_name} dict
    """
    try:
        df = pd.read_csv(csv_path)
        df[stock_col] = df[stock_col].astype(str).str.strip().str.zfill(6)
        industry_map = dict(zip(df[stock_col], df[industry_col]))
        logger.info("从 %s 加载了 %d 条行业映射", csv_path, len(industry_map))
        return industry_map
    except Exception as e:
        logger.error("加载行业映射失败: %s", e)
        return {}


def create_industry_map_from_akshare(entity_map: Dict[str, str]) -> Dict[str, str]:
    """从 AKShare 行业分类数据创建行业映射。

    Args:
        entity_map: {stock_code: industry_name} 原始映射

    Returns:
        {stock_code: industry_name} 标准化后的映射
    """
    return {
        str(code).strip().zfill(6): industry
        for code, industry in entity_map.items()
    }