"""
风控约束模块 - 交易前的风险控制检查

功能:
    - 单只股票最大仓位检查
    - 行业集中度检查
    - 止损触发检查
    - 最小持仓天数检查
    - 综合约束检查

使用方法:
    checker = RiskConstraints()
    violations = checker.apply_all_constraints(holdings_df, current_prices, constraints_config)
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


class RiskConstraints:
    """风险控制约束检查器。

    Attributes:
        default_max_position_pct: 默认单票最大仓位比例。
        default_max_industry_pct: 默认行业最大集中度。
        default_stop_loss_pct: 默认止损比例。
        default_min_hold_days: 默认最小持仓天数。
    """

    def __init__(
        self,
        default_max_position_pct: float = 0.10,
        default_max_industry_pct: float = 0.30,
        default_stop_loss_pct: float = 0.08,
        default_min_hold_days: int = 5,
    ):
        """初始化风控约束检查器。

        Args:
            default_max_position_pct: 默认单票最大仓位（占总资产比例）。
            default_max_industry_pct: 默认行业最大集中度。
            default_stop_loss_pct: 默认止损比例（亏损超过此比例强制卖出）。
            default_min_hold_days: 默认最小持仓天数。
        """
        self.default_max_position_pct = default_max_position_pct
        self.default_max_industry_pct = default_max_industry_pct
        self.default_stop_loss_pct = default_stop_loss_pct
        self.default_min_hold_days = default_min_hold_days

    def check_max_position(
        self,
        stocks_df: pd.DataFrame,
        max_pct: float = 0.10,
        weight_col: str = "weight",
    ) -> pd.DataFrame:
        """检查单只股票是否超过最大仓位限制。

        Args:
            stocks_df: 包含weight列的持仓DataFrame，必须包含stock_code列。
            max_pct: 最大仓位比例。
            weight_col: 权重列名。

        Returns:
            添加了'position_violation'列的DataFrame，
            True表示超过限制，False表示合规。
        """
        df = stocks_df.copy()

        if weight_col not in df.columns:
            logger.warning(f"未找到{weight_col}列，无法检查仓位限制")
            df["position_violation"] = False
            return df

        max_pct = max_pct or self.default_max_position_pct
        df["position_violation"] = df[weight_col] > max_pct

        violation_count = df["position_violation"].sum()
        if violation_count > 0:
            violators = df.loc[df["position_violation"], "stock_code"].tolist()
            logger.warning(
                f"仓位超限: {violation_count}只股票超过{max_pct:.1%}限制: {violators}"
            )
        else:
            logger.info(f"仓位检查通过: 所有股票权重 <= {max_pct:.1%}")

        return df

    def check_max_industry(
        self,
        stocks_df: pd.DataFrame,
        max_pct: float = 0.30,
        industry_col: str = "industry",
        weight_col: str = "weight",
    ) -> pd.DataFrame:
        """检查行业集中度是否超过限制。

        Args:
            stocks_df: 持仓DataFrame。
            max_pct: 行业最大集中度。
            industry_col: 行业列名。
            weight_col: 权重列名。

        Returns:
            添加了'industry_violation'、'industry_total_weight'列的DataFrame。
            同一行业的所有股票标记相同的违规状态。
        """
        df = stocks_df.copy()
        max_pct = max_pct or self.default_max_industry_pct

        if industry_col not in df.columns:
            logger.warning(f"无{industry_col}列，跳过行业集中度检查")
            df["industry_violation"] = False
            df["industry_total_weight"] = 0.0
            return df

        if weight_col not in df.columns:
            logger.warning(f"无{weight_col}列，跳过行业集中度检查")
            df["industry_violation"] = False
            df["industry_total_weight"] = 0.0
            return df

        # 计算每个行业的总权重
        industry_weights = df.groupby(industry_col)[weight_col].sum()
        df["industry_total_weight"] = df[industry_col].map(industry_weights)
        df["industry_violation"] = df["industry_total_weight"] > max_pct

        violations = industry_weights[industry_weights > max_pct]
        if len(violations) > 0:
            logger.warning(
                f"行业集中度超限: {len(violations)}个行业超过{max_pct:.1%}限制: "
                f"{dict(violations.round(4))}"
            )
        else:
            logger.info(f"行业集中度检查通过")

        return df

    def apply_stop_loss(
        self,
        holdings: pd.DataFrame,
        current_prices: Dict[str, float],
        entry_prices: Dict[str, float],
        stop_loss_pct: float = 0.08,
    ) -> pd.DataFrame:
        """止损检查 - 持仓亏损超过阈值时标记强制平仓。

        Args:
            holdings: 持仓DataFrame，必须包含stock_code列。
            current_prices: {stock_code: current_price} 当前价格。
            entry_prices: {stock_code: entry_price} 建仓价格。
            stop_loss_pct: 止损比例（正数，如0.08表示亏损8%）。

        Returns:
            添加了以下列的DataFrame:
            - pnl_pct: 盈亏比例
            - stop_loss_triggered: 是否触发止损
        """
        df = holdings.copy()
        stop_loss_pct = stop_loss_pct or self.default_stop_loss_pct

        pnl_pct_list = []
        triggered_list = []

        for _, row in df.iterrows():
            code = row["stock_code"]
            entry_price = entry_prices.get(code)
            current_price = current_prices.get(code)

            if entry_price is None or current_price is None or entry_price <= 0:
                pnl_pct_list.append(0.0)
                triggered_list.append(False)
                continue

            pnl_pct = (current_price - entry_price) / entry_price
            triggered = pnl_pct <= -stop_loss_pct

            pnl_pct_list.append(pnl_pct)
            triggered_list.append(triggered)

        df["pnl_pct"] = pnl_pct_list
        df["stop_loss_triggered"] = triggered_list

        triggered_count = sum(triggered_list)
        if triggered_count > 0:
            triggered_codes = df.loc[df["stop_loss_triggered"], "stock_code"].tolist()
            logger.warning(
                f"止损触发: {triggered_count}只股票跌破{-stop_loss_pct:.1%}: {triggered_codes}"
            )

        return df

    def check_min_hold_days(
        self,
        holdings: pd.DataFrame,
        min_days: int = 5,
        current_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """最小持仓天数检查 - 标记持仓不满min_days的股票不可卖出。

        Args:
            holdings: 持仓DataFrame，必须包含:
                - stock_code: 股票代码
                - entry_date: 建仓日期
            min_days: 最小持仓天数。
            current_date: 当前日期（YYYY-MM-DD），None则使用now。

        Returns:
            添加了以下列的DataFrame:
            - hold_days: 已持仓天数
            - sell_restricted: 是否限制卖出（持仓不足min_days）
        """
        df = holdings.copy()
        min_days = min_days or self.default_min_hold_days

        if "entry_date" not in df.columns:
            logger.warning("无entry_date列，跳过最小持仓天数检查")
            df["hold_days"] = 0
            df["sell_restricted"] = False
            return df

        if current_date is None:
            current_date = pd.Timestamp.now().strftime("%Y-%m-%d")

        current_dt = pd.Timestamp(current_date)

        hold_days_list = []
        for _, row in df.iterrows():
            entry_date = row.get("entry_date")
            if pd.isna(entry_date):
                hold_days_list.append(0)
                continue
            try:
                entry_dt = pd.Timestamp(entry_date)
                hold_days_list.append((current_dt - entry_dt).days)
            except Exception:
                hold_days_list.append(0)

        df["hold_days"] = hold_days_list
        df["sell_restricted"] = df["hold_days"] < min_days

        restricted_count = df["sell_restricted"].sum()
        if restricted_count > 0:
            logger.info(f"卖出限制: {restricted_count}只股票持仓不足{min_days}天")

        return df

    def apply_all_constraints(
        self,
        holdings_df: pd.DataFrame,
        current_prices: Optional[Dict[str, float]] = None,
        constraints_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """综合应用所有风控约束。

        Args:
            holdings_df: 持仓DataFrame，应包含stock_code, weight等列。
                如果有entry_price列则进行止损检查。
                如果有entry_date列则进行最小持仓天数检查。
            current_prices: {stock_code: price} 当前价格字典。
            constraints_config: 约束配置字典，可覆盖默认值:
                {
                    "max_position_pct": 0.10,
                    "max_industry_pct": 0.30,
                    "stop_loss_pct": 0.08,
                    "min_hold_days": 5,
                }

        Returns:
            (应用约束后的DataFrame, 违规摘要字典)
        """
        config = constraints_config or {}
        df = holdings_df.copy()

        max_position_pct = config.get("max_position_pct", self.default_max_position_pct)
        max_industry_pct = config.get("max_industry_pct", self.default_max_industry_pct)
        stop_loss_pct = config.get("stop_loss_pct", self.default_stop_loss_pct)
        min_hold_days = config.get("min_hold_days", self.default_min_hold_days)

        # 1. 单票仓位检查
        df = self.check_max_position(df, max_pct=max_position_pct)

        # 2. 行业集中度检查
        df = self.check_max_industry(df, max_pct=max_industry_pct)

        # 3. 止损检查
        if current_prices and "entry_price" in df.columns:
            entry_prices = {}
            for _, row in df.iterrows():
                code = row["stock_code"]
                ep = row.get("entry_price")
                if pd.notna(ep) and ep > 0:
                    entry_prices[code] = ep
            df = self.apply_stop_loss(
                df, current_prices, entry_prices, stop_loss_pct=stop_loss_pct
            )

        # 4. 最小持仓天数检查
        df = self.check_min_hold_days(df, min_days=min_hold_days)

        # 汇总违规
        summary = {
            "total_positions": len(df),
            "position_violations": int(df.get("position_violation", False).sum()),
            "industry_violations": int(df.get("industry_violation", False).sum()),
            "stop_loss_triggered": int(df.get("stop_loss_triggered", False).sum()),
            "sell_restricted": int(df.get("sell_restricted", False).sum()),
            "total_violations": (
                int(df.get("position_violation", False).sum())
                + int(df.get("industry_violation", False).sum())
                + int(df.get("stop_loss_triggered", False).sum())
                + int(df.get("sell_restricted", False).sum())
            ),
        }

        logger.info(f"风控约束检查完成: {summary}")
        return df, summary


class ConstraintsConfig:
    """风控约束配置类，集中管理所有约束参数。"""

    def __init__(
        self,
        max_position_pct: float = 0.10,
        max_industry_pct: float = 0.30,
        stop_loss_pct: float = 0.08,
        min_hold_days: int = 5,
        max_drawdown_pct: float = 0.20,
        max_leverage: float = 1.0,
        min_cash_ratio: float = 0.05,
    ):
        """初始化约束配置。

        Args:
            max_position_pct: 单票最大仓位。
            max_industry_pct: 行业最大集中度。
            stop_loss_pct: 止损比例。
            min_hold_days: 最小持仓天数。
            max_drawdown_pct: 最大回撤容忍度（触发减仓）。
            max_leverage: 最大杠杆倍数（1.0 = 无杠杆）。
            min_cash_ratio: 最低现金比例。
        """
        self.max_position_pct = max_position_pct
        self.max_industry_pct = max_industry_pct
        self.stop_loss_pct = stop_loss_pct
        self.min_hold_days = min_hold_days
        self.max_drawdown_pct = max_drawdown_pct
        self.max_leverage = max_leverage
        self.min_cash_ratio = min_cash_ratio

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典。"""
        return {
            "max_position_pct": self.max_position_pct,
            "max_industry_pct": self.max_industry_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "min_hold_days": self.min_hold_days,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_leverage": self.max_leverage,
            "min_cash_ratio": self.min_cash_ratio,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ConstraintsConfig":
        """从字典创建配置。"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__init__.__code__.co_varnames})