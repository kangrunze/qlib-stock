"""
真实交易约束模块 - 模拟A股市场的涨跌停、停牌、ST等约束

功能:
    - 涨停/跌停检测和处理
    - ST股票特殊涨跌幅限制（5%）
    - 停牌检测
    - 综合交易约束过滤

涨跌幅规则:
    - 普通A股: +10% / -10%
    - ST股票: +5% / -5%
    - 北交所: +30% / -30%
    - 科创板/创业板: +20% / -20%

使用方法:
    checker = RealisticConstraints()
    valid_orders = checker.apply_trading_constraints(order_list, price_data, date)
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)


class RealisticConstraints:
    """A股真实交易约束检查器。

    处理涨跌停板限制、ST股票限制、停牌检测等实际交易约束。

    Attributes:
        limit_up_ratio: 普通股涨停比例。
        limit_down_ratio: 普通股跌停比例。
        st_limit_ratio: ST股涨跌停比例。
        allow_st_trading: 是否允许交易ST股票。
    """

    # 各板块涨跌幅限制
    LIMIT_RATIOS = {
        "main": (0.10, -0.10),       # 主板
        "gem": (0.20, -0.20),         # 创业板
        "star": (0.20, -0.20),        # 科创板
        "bse": (0.30, -0.30),         # 北交所
        "st": (0.05, -0.05),           # ST
    }

    def __init__(
        self,
        limit_up_ratio: float = 0.10,
        limit_down_ratio: float = 0.10,
        st_limit_ratio: float = 0.05,
        allow_st_trading: bool = False,
        board_type: str = "main",
    ):
        """初始化交易约束检查器。

        Args:
            limit_up_ratio: 普通股涨停比例（正数）。
            limit_down_ratio: 普通股跌停比例（正数）。
            st_limit_ratio: ST股涨跌停比例（正数）。
            allow_st_trading: 是否允许交易ST股票。
            board_type: 默认板块类型 (main/gem/star/bse)。
        """
        self.limit_up_ratio = limit_up_ratio
        self.limit_down_ratio = limit_down_ratio
        self.st_limit_ratio = st_limit_ratio
        self.allow_st_trading = allow_st_trading
        self.board_type = board_type

    def _get_limit_ratios(self, is_st: bool, board_type: Optional[str] = None) -> Tuple[float, float]:
        """获取涨跌停比例。

        Args:
            is_st: 是否为ST股票。
            board_type: 板块类型。

        Returns:
            (涨停比例, 跌停比例) 均为正数。
        """
        if is_st:
            return self.st_limit_ratio, self.st_limit_ratio

        bt = board_type or self.board_type
        return self.LIMIT_RATIOS.get(bt, (self.limit_up_ratio, self.limit_down_ratio))

    def is_limit_up(
        self,
        price: float,
        preclose: float,
        is_st: bool = False,
        board_type: Optional[str] = None,
    ) -> bool:
        """检查是否涨停。

        涨停条件：当前价 >= 涨停价 = preclose * (1 + limit_up_ratio)。
        使用近似比较容忍小误差（1e-6）。

        Args:
            price: 当前价格。
            preclose: 前收盘价。
            is_st: 是否为ST股票。
            board_type: 板块类型。

        Returns:
            是否涨停。
        """
        if preclose <= 0 or price <= 0:
            return False

        up_ratio, _ = self._get_limit_ratios(is_st, board_type)
        limit_price = preclose * (1 + up_ratio)
        return price >= limit_price - 1e-6

    def is_limit_down(
        self,
        price: float,
        preclose: float,
        is_st: bool = False,
        board_type: Optional[str] = None,
    ) -> bool:
        """检查是否跌停。

        跌停条件：当前价 <= 跌停价 = preclose * (1 - limit_down_ratio)。

        Args:
            price: 当前价格。
            preclose: 前收盘价。
            is_st: 是否为ST股票。
            board_type: 板块类型。

        Returns:
            是否跌停。
        """
        if preclose <= 0 or price <= 0:
            return False

        _, down_ratio = self._get_limit_ratios(is_st, board_type)
        limit_price = preclose * (1 - down_ratio)
        return price <= limit_price + 1e-6

    def is_suspended(self, volume: float) -> bool:
        """检查是否停牌。

        停牌判断：成交量为0（忽略价格是否有变动）。

        Args:
            volume: 成交量（手或股）。

        Returns:
            是否停牌。
        """
        return volume is None or pd.isna(volume) or volume <= 0

    def can_buy(
        self,
        price: float,
        preclose: float,
        volume: float,
        is_st: bool = False,
        board_type: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """检查是否可以买入。

        不可买入的情况：
        1. 涨停 - 买不到
        2. 停牌 - 无法交易
        3. ST股票且不允许交易ST

        Args:
            price: 当前价格。
            preclose: 前收盘价。
            volume: 成交量。
            is_st: 是否ST。
            board_type: 板块类型。

        Returns:
            (是否可买, 原因说明)
        """
        if is_st and not self.allow_st_trading:
            return False, "ST股票不允许交易"

        if self.is_suspended(volume):
            return False, "停牌"

        if self.is_limit_up(price, preclose, is_st, board_type):
            return False, "涨停"

        return True, "可买入"

    def can_sell(
        self,
        price: float,
        preclose: float,
        volume: float,
        is_st: bool = False,
        board_type: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """检查是否可以卖出。

        不可卖出的情况：
        1. 跌停 - 卖不出去（非ST股票可以排队但回测简化处理）
        2. 停牌 - 无法交易

        注意：跌停时理论上可以挂单排队，但回测中保守处理为不可卖出。

        Args:
            price: 当前价格。
            preclose: 前收盘价。
            volume: 成交量。
            is_st: 是否ST。
            board_type: 板块类型。

        Returns:
            (是否可卖, 原因说明)
        """
        if self.is_suspended(volume):
            return False, "停牌"

        if self.is_limit_down(price, preclose, is_st, board_type):
            return False, "跌停"

        return True, "可卖出"

    def apply_trading_constraints(
        self,
        order_list: List[Dict[str, Any]],
        price_data: pd.DataFrame,
        date: str,
    ) -> List[Dict[str, Any]]:
        """对订单列表应用交易约束，过滤不可执行的订单。

        Args:
            order_list: 订单列表，每个订单为字典:
                {
                    "stock_code": str,
                    "side": "buy" | "sell",
                    "shares": int,
                    "price": float (可选, 从price_data取),
                }
            price_data: 当天价格数据，包含列:
                - stock_code (或作为index)
                - close: 收盘价
                - preclose: 前收盘价
                - volume: 成交量
            date: 交易日期。

        Returns:
            过滤后可执行的订单列表，每个订单增加 'executable' 和 'reason' 字段。
        """
        # 确保以stock_code索引
        if "stock_code" in price_data.columns:
            price_lookup = price_data.set_index("stock_code")
        else:
            price_lookup = price_data.copy()

        valid_orders = []
        filter_stats = {"total": len(order_list), "passed": 0, "rejected": 0, "reasons": {}}

        for order in order_list:
            code = order["stock_code"]
            side = order.get("side", "buy")

            # 从价格数据获取信息
            if code not in price_lookup.index:
                order["executable"] = False
                order["reason"] = f"无价格数据"
                filter_stats["rejected"] += 1
                filter_stats["reasons"]["no_price_data"] = filter_stats["reasons"].get("no_price_data", 0) + 1
                valid_orders.append(order)
                continue

            row = price_lookup.loc[code]

            # 处理多行情况
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            close_price = order.get("price", row.get("close", 0))
            preclose = row.get("preclose", close_price)
            volume = row.get("volume", 0)
            is_st = row.get("is_st", False)

            if side == "buy":
                can_exec, reason = self.can_buy(close_price, preclose, volume, is_st)
            else:
                can_exec, reason = self.can_sell(close_price, preclose, volume, is_st)

            order["executable"] = can_exec
            order["reason"] = reason
            order["price"] = close_price

            if can_exec:
                filter_stats["passed"] += 1
            else:
                filter_stats["rejected"] += 1
                filter_stats["reasons"][reason] = filter_stats["reasons"].get(reason, 0) + 1

            valid_orders.append(order)

        logger.info(
            f"交易约束过滤 [{date}]: {filter_stats['total']}笔订单 -> "
            f"通过{filter_stats['passed']}笔, 拒绝{filter_stats['rejected']}笔 "
            f"(原因: {filter_stats['reasons']})"
        )
        return valid_orders


def detect_market_conditions(
    price_data: pd.DataFrame,
    date: str,
) -> Dict[str, Any]:
    """检测市场整体状况。

    包括：涨停家数、跌停家数、停牌家数等统计。

    Args:
        price_data: 当日价格数据。
        date: 日期。

    Returns:
        市场状况统计字典。
    """
    constraints = RealisticConstraints()
    stats = {
        "date": date,
        "total_stocks": len(price_data),
        "limit_up_count": 0,
        "limit_down_count": 0,
        "suspended_count": 0,
        "tradable_count": 0,
    }

    for _, row in price_data.iterrows():
        close = row.get("close", 0)
        preclose = row.get("preclose", close)
        volume = row.get("volume", 0)
        is_st = row.get("is_st", False)

        if constraints.is_limit_up(close, preclose, is_st):
            stats["limit_up_count"] += 1
        if constraints.is_limit_down(close, preclose, is_st):
            stats["limit_down_count"] += 1
        if constraints.is_suspended(volume):
            stats["suspended_count"] += 1

    stats["tradable_count"] = (
        stats["total_stocks"]
        - stats["limit_up_count"]
        - stats["limit_down_count"]
        - stats["suspended_count"]
    )

    # 计算市场极端程度
    stats["limit_up_ratio"] = stats["limit_up_count"] / max(stats["total_stocks"], 1)
    stats["limit_down_ratio"] = stats["limit_down_count"] / max(stats["total_stocks"], 1)

    logger.info(
        f"市场状况 [{date}]: 涨停{stats['limit_up_count']}, "
        f"跌停{stats['limit_down_count']}, 停牌{stats['suspended_count']}, "
        f"可交易{stats['tradable_count']}/{stats['total_stocks']}"
    )
    return stats