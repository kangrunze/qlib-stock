"""
回测引擎模块 - 完整的A股量化回测模拟器

功能:
    - 模拟多期调仓交易
    - 追踪现金、持仓、权益曲线、交易记录
    - 处理退市、停牌、涨跌停等A股特有约束
    - 计算滑点和手续费
    - 支持每日盯市（mark-to-market）
    - 生成完整业绩报告

使用方法:
    engine = BacktestEngine(initial_cash=1000000, commission_buy=0.0003)
    results = engine.run(price_data, predictions, strategy, start_date, end_date)
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

# 尝试导入同级模块
try:
    from .realistic_constraints import RealisticConstraints
    from .performance_report import PerformanceReport
    from .benchmark_loader import BenchmarkLoader, compare_to_benchmark
except ImportError:
    from backtest.realistic_constraints import RealisticConstraints
    from backtest.performance_report import PerformanceReport
    from backtest.benchmark_loader import BenchmarkLoader, compare_to_benchmark


class BacktestEngine:
    """完整回测引擎。

    模拟在给定日期范围内的多期调仓交易，追踪所有交易细节和资产变化。

    Attributes:
        initial_cash: 初始资金。
        commission_buy: 买入手续费率。
        commission_sell: 卖出手续费率（含印花税）。
        slippage: 滑点比例。
        benchmark_code: 基准指数代码。
        min_commission: 最低手续费（元）。
        stamp_duty: 印花税率（卖出时收取）。
    """

    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        commission_buy: float = 0.0003,       # 万三佣金
        commission_sell: float = 0.0013,       # 万三佣金 + 千一印花税
        slippage: float = 0.001,               # 0.1% 滑点
        benchmark_code: str = "000905",
        min_commission: float = 5.0,           # 最低5元佣金
    ):
        """初始化回测引擎。

        Args:
            initial_cash: 初始资金（元）。
            commission_buy: 买入手续费率。
            commission_sell: 卖出手续费率（通常含印花税0.1%）。
            slippage: 滑点比例，模拟成交价偏离。
            benchmark_code: 基准指数代码（如000905中证500）。
            min_commission: 最低手续费（元），A股通常5元。
        """
        self.initial_cash = initial_cash
        self.commission_buy = commission_buy
        self.commission_sell = commission_sell
        self.slippage = slippage
        self.benchmark_code = benchmark_code
        self.min_commission = min_commission

        # 重置状态
        self._reset()

    def _reset(self) -> None:
        """重置回测状态到初始值。"""
        self.cash = self.initial_cash
        self.holdings: Dict[str, Dict[str, Any]] = {}  # {stock_code: {shares, avg_cost, entry_date}}
        self.equity_curve: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []
        self.daily_positions: List[Dict[str, Any]] = []
        self.trade_id_counter = 0

    def run(
        self,
        price_data_dict: Dict[str, pd.DataFrame],
        predictions_by_date: Dict[str, pd.DataFrame],
        strategy: Any,
        start_date: str,
        end_date: str,
        rebalance_freq: str = "monthly",
        top_k: int = 30,
        weight_method: str = "equal",
        use_constraints: bool = True,
        benchmark_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """执行回测。

        Args:
            price_data_dict: {stock_code: DataFrame} 价格数据字典。
                每个DataFrame包含: date(index), open, high, low, close, volume, preclose。
            predictions_by_date: {date_str: predictions_DataFrame} 每日预测数据。
                每个DataFrame包含: stock_code, score, industry, is_st。
            strategy: 策略对象（如TopKSelector实例），必须实现select方法。
            start_date: 回测起始日期 (YYYY-MM-DD)。
            end_date: 回测结束日期 (YYYY-MM-DD)。
            rebalance_freq: 调仓频率 ("daily", "weekly", "monthly")。
            top_k: 每次选股数量。
            weight_method: 权重分配方法 ("equal", "score_weighted")。
            use_constraints: 是否使用真实交易约束（涨跌停、停牌）。
            benchmark_data: 基准指数净值数据（可选）。

        Returns:
            包含以下内容的字典:
            - equity_curve: 每日权益DataFrame
            - trades: 交易明细DataFrame
            - metrics: 业绩指标字典
            - benchmark_comparison: 基准对比结果
        """
        logger.info("=" * 60)
        logger.info(f"开始回测: {start_date} ~ {end_date}")
        logger.info(f"初始资金: {self.initial_cash:,.0f}, 调仓频率: {rebalance_freq}")
        logger.info("=" * 60)

        self._reset()

        # 获取调仓日期
        rebalance_dates = self._get_rebalance_dates(start_date, end_date, rebalance_freq)
        logger.info(f"调仓日期数: {len(rebalance_dates)}")

        if not rebalance_dates:
            logger.warning("无有效调仓日期，返回空结果")
            return self._empty_result()

        # 获取所有交易日
        all_dates = self._get_all_trading_dates(price_data_dict, start_date, end_date)
        logger.info(f"总交易日数: {len(all_dates)}")

        # 构建交易约束检查器
        constraints = RealisticConstraints() if use_constraints else None

        # 逐日模拟
        next_rebalance_idx = 0
        current_target_stocks: set = set()

        for date_idx, date in enumerate(all_dates):
            date_str = str(date.date()) if hasattr(date, 'date') else str(date)[:10]

            # 检查是否调仓日
            is_rebalance = (
                next_rebalance_idx < len(rebalance_dates)
                and date_str >= rebalance_dates[next_rebalance_idx]
            )

            if is_rebalance:
                # 使用最近的调仓日（跳过非交易日）
                reb_date = rebalance_dates[next_rebalance_idx]
                next_rebalance_idx += 1

                # 执行调仓
                self._rebalance(
                    date_str=date_str,
                    price_data_dict=price_data_dict,
                    predictions_by_date=predictions_by_date,
                    strategy=strategy,
                    top_k=top_k,
                    weight_method=weight_method,
                    constraints=constraints,
                )

            # 每日盯市
            self._mark_to_market(date_str, price_data_dict)

        # 最后一天强制平仓（计算最终价值）
        self._final_liquidate(all_dates[-1], price_data_dict)

        # 生成结果
        results = self._build_results(all_dates, benchmark_data, start_date, end_date)

        logger.info("=" * 60)
        logger.info("回测完成")
        logger.info(f"最终权益: {self._get_total_equity(all_dates[-1], price_data_dict):,.2f}")
        metrics = results.get("metrics", {})
        logger.info(f"累计收益: {metrics.get('total_return_pct', 0):.2f}%")
        logger.info(f"Sharpe比率: {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info("=" * 60)

        return results

    def _get_rebalance_dates(
        self, start_date: str, end_date: str, frequency: str
    ) -> List[str]:
        """生成调仓日期列表。

        Args:
            start_date: 起始日期。
            end_date: 结束日期。
            frequency: 调仓频率 ("daily", "weekly", "monthly")。

        Returns:
            调仓日期字符串列表。
        """
        freq_map = {
            "daily": "B",
            "weekly": "W-FRI",
            "monthly": "BMS",  # 月初第一个工作日
            "month_end": "BM",  # 月末最后一个工作日
        }

        freq = freq_map.get(frequency, "BMS")

        dates = pd.date_range(start_date, end_date, freq=freq)
        return [str(d.date()) for d in dates]

    def _get_all_trading_dates(
        self,
        price_data_dict: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
    ) -> pd.DatetimeIndex:
        """从价格数据中提取所有交易日。

        取所有股票日期并集的排序后列表。

        Args:
            price_data_dict: 价格数据。
            start_date: 起始日期。
            end_date: 结束日期。

        Returns:
            排序后的交易日DatetimeIndex。
        """
        all_dates_set = set()
        for code, df in price_data_dict.items():
            if df is not None and not df.empty:
                idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
                mask = (idx >= start_date) & (idx <= end_date)
                all_dates_set.update(idx[mask])

        all_dates = sorted(all_dates_set)
        return pd.DatetimeIndex(all_dates)

    def _rebalance(
        self,
        date_str: str,
        price_data_dict: Dict[str, pd.DataFrame],
        predictions_by_date: Dict[str, pd.DataFrame],
        strategy: Any,
        top_k: int,
        weight_method: str,
        constraints: Optional[RealisticConstraints],
    ) -> None:
        """执行一期调仓。

        Args:
            date_str: 调仓日期。
            price_data_dict: 价格数据。
            predictions_by_date: 预测数据。
            strategy: 策略对象。
            top_k: 选股数。
            weight_method: 权重方法。
            constraints: 交易约束检查器。
        """
        logger.info(f"\n--- 调仓: {date_str} ---")

        # 获取当日价格快照
        price_snapshot = self._get_price_snapshot(date_str, price_data_dict)
        if not price_snapshot:
            logger.warning(f"{date_str} 无可用价格数据，跳过调仓")
            return

        # 获取当日预测
        predictions = self._get_predictions(date_str, predictions_by_date)
        if predictions is None or predictions.empty:
            logger.warning(f"{date_str} 无预测数据，保持当前持仓")
            return

        # 执行选股
        try:
            selected = strategy.select(predictions, date_str, top_k=top_k)
        except Exception as e:
            logger.error(f"选股失败: {e}")
            return

        if selected.empty:
            logger.warning(f"{date_str} 选股结果为空")
            return

        new_stock_codes = set(selected["stock_code"].tolist())
        current_stock_codes = set(self.holdings.keys())

        # 计算权重
        n_stocks = len(selected)
        selected["weight"] = 1.0 / n_stocks if weight_method == "equal" else selected["weight"]

        # ===== 卖出不在新选股中的持仓 =====
        stocks_to_sell = current_stock_codes - new_stock_codes
        for code in stocks_to_sell:
            if code not in self.holdings:
                continue
            holding = self.holdings[code]
            shares = holding["shares"]
            if shares <= 0:
                continue

            # 获取卖出价格
            price_info = price_snapshot.get(code, {})
            sell_price = price_info.get("close", 0)
            preclose = price_info.get("preclose", sell_price)
            volume = price_info.get("volume", 0)
            is_st = holding.get("is_st", False)

            # 检查约束
            if constraints:
                can_sell, reason = constraints.can_sell(sell_price, preclose, volume, is_st)
                if not can_sell:
                    logger.info(f"  {code} 无法卖出: {reason}, 冻结持仓")
                    continue

            # 执行卖出
            proceeds = self._execute_sell(date_str, code, shares, sell_price)
            logger.info(f"  卖出 {code}: {shares}股 @ {sell_price:.2f}, 收入={proceeds:,.2f}")

        # ===== 买入新选股 =====
        stocks_to_buy = new_stock_codes - current_stock_codes
        if stocks_to_buy:
            total_capital = self._get_total_equity(date_str, price_data_dict)
            cash_per_new = (total_capital * 0.95 / max(len(stocks_to_buy), 1))  # 留5%现金缓冲

            for code in stocks_to_buy:
                if code == "cash":
                    continue

                price_info = price_snapshot.get(code, {})
                buy_price = price_info.get("close", 0)
                preclose = price_info.get("preclose", buy_price)
                volume = price_info.get("volume", 0)
                is_st = price_info.get("is_st", False)

                if buy_price <= 0:
                    logger.warning(f"  {code} 无有效价格，跳过买入")
                    continue

                # 检查约束
                if constraints:
                    can_buy, reason = constraints.can_buy(buy_price, preclose, volume, is_st)
                    if not can_buy:
                        logger.info(f"  {code} 无法买入: {reason}")
                        continue

                # 检查资金
                if self.cash < buy_price * 100:  # 至少够买1手
                    logger.info(f"  {code} 资金不足，跳过买入 (现金: {self.cash:,.0f})")
                    continue

                target_amount = min(cash_per_new, self.cash * 0.95)
                shares_bought = self._execute_buy(date_str, code, target_amount, buy_price)

                if shares_bought > 0:
                    # 记录持仓
                    self.holdings[code] = {
                        "shares": shares_bought,
                        "avg_cost": buy_price,
                        "entry_date": date_str,
                        "is_st": is_st,
                        "industry": price_info.get("industry", ""),
                    }
                    logger.info(f"  买入 {code}: {shares_bought}股 @ {buy_price:.2f}")

        # 更新已有持仓的权重（如果需要等权再平衡）
        # 暂不实现再平衡，保持简单

    def _get_price_snapshot(
        self, date_str: str, price_data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, Any]]:
        """获取某日所有股票的价格快照。

        Args:
            date_str: 日期。
            price_data_dict: 价格数据字典。

        Returns:
            {stock_code: {close, open, volume, preclose, is_st, industry}} 字典。
        """
        snapshot = {}
        target_date = pd.Timestamp(date_str)

        for code, df in price_data_dict.items():
            if df is None or df.empty:
                continue

            try:
                if target_date in df.index:
                    row = df.loc[target_date]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]

                    snapshot[code] = {
                        "close": float(row.get("close", 0)),
                        "open": float(row.get("open", row.get("close", 0))),
                        "volume": float(row.get("volume", 0)),
                        "preclose": float(row.get("preclose", row.get("close", 0))),
                        "is_st": bool(row.get("is_st", False)),
                        "industry": str(row.get("industry", "")),
                    }
            except Exception:
                continue

        return snapshot

    def _get_predictions(
        self, date_str: str, predictions_by_date: Dict[str, pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """获取某日预测数据。

        Args:
            date_str: 日期。
            predictions_by_date: {date: DataFrame} 预测数据。

        Returns:
            预测DataFrame或None。
        """
        if date_str in predictions_by_date:
            return predictions_by_date[date_str]

        # 寻找最近的预测日期
        available_dates = sorted(predictions_by_date.keys())
        for d in reversed(available_dates):
            if d <= date_str:
                logger.info(f"使用最近预测日期: {d} (请求: {date_str})")
                return predictions_by_date[d]

        return None

    def _execute_sell(
        self, date: str, stock_code: str, shares: int, price: float
    ) -> float:
        """执行卖出操作。

        Args:
            date: 交易日期。
            stock_code: 股票代码。
            shares: 卖出股数。
            price: 卖出价格。

        Returns:
            扣除费用后的实际收入。
        """
        if shares <= 0 or price <= 0:
            return 0.0

        # 应用滑点（卖出时价格偏低）
        execution_price = price * (1 - self.slippage)

        gross_proceeds = shares * execution_price

        # 计算手续费（佣金 + 印花税）
        commission = max(gross_proceeds * self.commission_sell, self.min_commission)
        net_proceeds = gross_proceeds - commission

        # 计算盈亏
        holding = self.holdings.get(stock_code, {})
        avg_cost = holding.get("avg_cost", price)
        pnl = (execution_price - avg_cost) * shares - commission

        # 更新现金
        self.cash += net_proceeds

        # 记录交易
        self.trade_id_counter += 1
        trade_record = {
            "trade_id": self.trade_id_counter,
            "date": date,
            "stock_code": stock_code,
            "side": "sell",
            "shares": shares,
            "price": execution_price,
            "gross_amount": gross_proceeds,
            "commission": commission,
            "net_amount": net_proceeds,
            "pnl": pnl,
            "pnl_pct": pnl / (avg_cost * shares) if avg_cost > 0 and shares > 0 else 0.0,
        }
        self.trades.append(trade_record)

        # 清除持仓
        if stock_code in self.holdings:
            del self.holdings[stock_code]

        return net_proceeds

    def _execute_buy(
        self, date: str, stock_code: str, target_amount: float, price: float
    ) -> int:
        """执行买入操作。

        Args:
            date: 交易日期。
            stock_code: 股票代码。
            target_amount: 目标买入金额。
            price: 买入价格。

        Returns:
            实际买入股数。
        """
        if target_amount <= 0 or price <= 0:
            return 0

        # 应用滑点（买入时价格偏高）
        execution_price = price * (1 + self.slippage)

        # 计算可买股数（A股以100股为1手）
        raw_shares = int(target_amount / execution_price / 100) * 100
        if raw_shares < 100:
            return 0

        # 检查资金是否足够
        gross_amount = raw_shares * execution_price
        commission = max(gross_amount * self.commission_buy, self.min_commission)
        total_cost = gross_amount + commission

        if total_cost > self.cash:
            # 减少买入量
            adjusted_shares = int((self.cash - self.min_commission) / execution_price / 100) * 100
            if adjusted_shares < 100:
                return 0
            raw_shares = adjusted_shares
            gross_amount = raw_shares * execution_price
            commission = max(gross_amount * self.commission_buy, self.min_commission)
            total_cost = gross_amount + commission

        if total_cost > self.cash:
            return 0

        # 更新现金
        self.cash -= total_cost

        # 更新持仓
        if stock_code in self.holdings:
            # 加仓 - 更新平均成本
            old_shares = self.holdings[stock_code]["shares"]
            old_cost = self.holdings[stock_code]["avg_cost"]
            new_shares = old_shares + raw_shares
            new_cost = (old_cost * old_shares + execution_price * raw_shares) / new_shares
            self.holdings[stock_code]["shares"] = new_shares
            self.holdings[stock_code]["avg_cost"] = new_cost
        else:
            self.holdings[stock_code] = {
                "shares": raw_shares,
                "avg_cost": execution_price,
                "entry_date": date,
            }

        # 记录交易
        self.trade_id_counter += 1
        trade_record = {
            "trade_id": self.trade_id_counter,
            "date": date,
            "stock_code": stock_code,
            "side": "buy",
            "shares": raw_shares,
            "price": execution_price,
            "gross_amount": gross_amount,
            "commission": commission,
            "net_amount": -total_cost,
            "pnl": 0.0,
            "pnl_pct": 0.0,
        }
        self.trades.append(trade_record)

        return raw_shares

    def _mark_to_market(
        self, date_str: str, price_data_dict: Dict[str, pd.DataFrame]
    ) -> None:
        """每日盯市，计算持仓市值和总权益。

        Args:
            date_str: 日期。
            price_data_dict: 价格数据。
        """
        snapshot = self._get_price_snapshot(date_str, price_data_dict)

        holdings_value = 0.0
        positions_detail = []

        for code, holding in self.holdings.items():
            shares = holding["shares"]
            if code in snapshot:
                close_price = snapshot[code]["close"]
                market_value = shares * close_price
            else:
                # 无价格数据，使用最近成本价
                market_value = shares * holding.get("avg_cost", 0)

            holdings_value += market_value
            positions_detail.append({
                "stock_code": code,
                "shares": shares,
                "market_value": market_value,
                "avg_cost": holding.get("avg_cost", 0),
            })

        total_equity = self.cash + holdings_value

        self.equity_curve.append({
            "date": date_str,
            "cash": self.cash,
            "holdings_value": holdings_value,
            "total_equity": total_equity,
            "num_positions": len(self.holdings),
        })

        self.daily_positions.append({
            "date": date_str,
            "positions": positions_detail,
        })

    def _final_liquidate(
        self, last_date: Any, price_data_dict: Dict[str, pd.DataFrame]
    ) -> None:
        """最后一天全部平仓，计算最终价值。

        Args:
            last_date: 最后日期。
            price_data_dict: 价格数据。
        """
        date_str = str(last_date.date()) if hasattr(last_date, 'date') else str(last_date)[:10]

        for code in list(self.holdings.keys()):
            holding = self.holdings[code]
            shares = holding["shares"]
            if shares <= 0:
                continue

            price_info = self._get_price_snapshot(date_str, price_data_dict).get(code, {})
            sell_price = price_info.get("close", holding.get("avg_cost", 0))

            if sell_price > 0:
                self._execute_sell(date_str, code, shares, sell_price)

    def _get_total_equity(
        self, date_str: str, price_data_dict: Dict[str, pd.DataFrame]
    ) -> float:
        """计算某日总权益（现金 + 持仓市值）。

        Args:
            date_str: 日期。
            price_data_dict: 价格数据。

        Returns:
            总权益。
        """
        snapshot = self._get_price_snapshot(date_str, price_data_dict)
        holdings_value = 0.0

        for code, holding in self.holdings.items():
            if code in snapshot:
                holdings_value += holding["shares"] * snapshot[code]["close"]
            else:
                holdings_value += holding["shares"] * holding.get("avg_cost", 0)

        return self.cash + holdings_value

    def _build_results(
        self,
        all_dates: pd.DatetimeIndex,
        benchmark_data: Optional[pd.DataFrame],
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        """构建回测结果。

        Args:
            all_dates: 所有交易日。
            benchmark_data: 基准数据。
            start_date: 起始日期。
            end_date: 结束日期。

        Returns:
            回测结果字典。
        """
        # 构建权益曲线DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        if not equity_df.empty:
            equity_df["date"] = pd.to_datetime(equity_df["date"])
            equity_df = equity_df.set_index("date")
            equity_df["equity"] = equity_df["total_equity"]
            equity_df["daily_return"] = equity_df["equity"].pct_change()

        # 构建交易明细DataFrame
        trades_df = pd.DataFrame(self.trades)

        # 计算业绩指标
        reporter = PerformanceReport()
        if not equity_df.empty:
            metrics = reporter.generate_report(
                equity_curve=equity_df,
                trades=trades_df,
                benchmark_equity=benchmark_data,
                start_date=start_date,
                end_date=end_date,
            )
        else:
            metrics = reporter._empty_report()

        # 基准对比
        benchmark_comparison = {}
        if benchmark_data is not None and not equity_df.empty:
            benchmark_comparison = compare_to_benchmark(
                equity_df["equity"], benchmark_data["equity"]
                if "equity" in benchmark_data.columns
                else benchmark_data.iloc[:, 0]
            )

        return {
            "equity_curve": equity_df,
            "trades": trades_df,
            "metrics": metrics,
            "benchmark_comparison": benchmark_comparison,
            "daily_positions": self.daily_positions,
            "config": {
                "initial_cash": self.initial_cash,
                "start_date": start_date,
                "end_date": end_date,
                "commission_buy": self.commission_buy,
                "commission_sell": self.commission_sell,
                "slippage": self.slippage,
                "benchmark_code": self.benchmark_code,
            },
        }

    def _empty_result(self) -> Dict[str, Any]:
        """返回空回测结果。"""
        return {
            "equity_curve": pd.DataFrame(),
            "trades": pd.DataFrame(),
            "metrics": PerformanceReport()._empty_report(),
            "benchmark_comparison": {},
            "daily_positions": [],
            "config": {
                "initial_cash": self.initial_cash,
                "commission_buy": self.commission_buy,
                "commission_sell": self.commission_sell,
                "slippage": self.slippage,
                "benchmark_code": self.benchmark_code,
            },
        }

    def compute_metrics(self) -> Dict[str, Any]:
        """从当前回测状态计算业绩指标。

        Returns:
            业绩指标字典。
        """
        if not self.equity_curve:
            return PerformanceReport()._empty_report()

        equity_df = pd.DataFrame(self.equity_curve)
        equity_df["date"] = pd.to_datetime(equity_df["date"])
        equity_df = equity_df.set_index("date")

        trades_df = pd.DataFrame(self.trades)

        reporter = PerformanceReport()
        return reporter.generate_report(equity_curve=equity_df, trades=trades_df)


def run_simple_backtest(
    price_data_dict: Dict[str, pd.DataFrame],
    predictions_by_date: Dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
    initial_cash: float = 1_000_000,
    top_k: int = 30,
    rebalance_freq: str = "monthly",
) -> Dict[str, Any]:
    """快速运行简单回测的便捷函数。

    使用默认策略和等权配置。

    Args:
        price_data_dict: 价格数据。
        predictions_by_date: 预测数据。
        start_date: 起始日期。
        end_date: 结束日期。
        initial_cash: 初始资金。
        top_k: 选股数。
        rebalance_freq: 调仓频率。

    Returns:
        回测结果字典。
    """
    from strategy.topk_selector import TopKSelector

    engine = BacktestEngine(initial_cash=initial_cash)
    strategy = TopKSelector(exclude_st=True, exclude_suspended=True)

    results = engine.run(
        price_data_dict=price_data_dict,
        predictions_by_date=predictions_by_date,
        strategy=strategy,
        start_date=start_date,
        end_date=end_date,
        rebalance_freq=rebalance_freq,
        top_k=top_k,
        weight_method="equal",
        use_constraints=True,
    )

    return results


if __name__ == "__main__":
    # 独立运行示例
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    print("BacktestEngine 模块加载成功。")
    print("使用方法:")
    print("  engine = BacktestEngine(initial_cash=1000000)")
    print("  results = engine.run(price_data_dict, predictions, strategy, start, end)")