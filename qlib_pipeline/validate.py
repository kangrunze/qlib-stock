# -*- coding: utf-8 -*-
"""推荐验证器 - 验证推荐股票的未来收益"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RecommendationValidator:
    """推荐验证器 - 计算推荐股票的后续收益"""

    def __init__(self, data_provider=None):
        self.data_provider = data_provider
        self.validation_results = []

    def validate_recommendations(
        self,
        recommendations_df: pd.DataFrame,
        price_data: pd.DataFrame,
        horizons: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """验证推荐股票的后续收益。

        2026-07-10 修复前视偏差:
          信号在 T 日收盘后计算（用 T 日的 close/成交量等特征），
          现实中最早只能在 T+1 日下单。买入价取 T+1 日开盘价，
          与正式回测的 deal_price: "open" 保持一致。
          收益窗口也从 T+1 开盘价算起，到 T+1+horizon 日开盘价。
        """
        results = []
        for _, rec in recommendations_df.iterrows():
            rec_date = rec["date"]
            stock = rec["stock_code"]
            rank = rec.get("rank", 0)
            score = rec.get("score", 0.0)
            try:
                if isinstance(price_data.index, pd.MultiIndex):
                    stock_prices = price_data.loc[stock]
                else:
                    stock_prices = price_data
                rec_date_idx = stock_prices.index.get_loc(rec_date)
                if isinstance(rec_date_idx, slice):
                    rec_date_idx = rec_date_idx.start

                # T+1 日为实际买入日（与回测 deal_price: open 一致）
                buy_idx = rec_date_idx + 1
                if buy_idx >= len(stock_prices):
                    logger.warning("验证失败 %s %s: T+1 日数据不存在", stock, rec_date)
                    continue

                result = {
                    "date": rec_date, "stock_code": stock,
                    "rank": rank, "score": score,
                    # 买入价 = T+1 日开盘价（无前视偏差）
                    "rec_price": stock_prices.iloc[buy_idx]["open"]
                }
                for horizon in horizons:
                    # 收益窗口: T+1 开盘 → T+1+horizon 开盘
                    future_idx = buy_idx + horizon
                    if future_idx < len(stock_prices):
                        future_price = stock_prices.iloc[future_idx]["open"]
                        ret = (future_price - result["rec_price"]) / result["rec_price"]
                        result[f"ret_{horizon}d"] = ret
                    else:
                        result[f"ret_{horizon}d"] = np.nan
                results.append(result)
            except Exception as e:
                logger.warning("验证失败 %s %s: %s", stock, rec_date, e)
                continue
        return pd.DataFrame(results)

    def calculate_hit_rate(
        self, validation_df: pd.DataFrame, horizon: int = 5, threshold: float = 0.0
    ) -> Dict:
        col = f"ret_{horizon}d"
        if col not in validation_df.columns:
            return {"hit_rate": 0.0, "avg_return": 0.0, "count": 0}
        valid_df = validation_df.dropna(subset=[col])
        if valid_df.empty:
            return {"hit_rate": 0.0, "avg_return": 0.0, "count": 0}
        hits = (valid_df[col] > threshold).sum()
        total = len(valid_df)
        return {
            "hit_rate": hits / total if total > 0 else 0.0,
            "avg_return": valid_df[col].mean(),
            "median_return": valid_df[col].median(),
            "max_return": valid_df[col].max(),
            "min_return": valid_df[col].min(),
            "count": total, "hits": int(hits)
        }

    def calculate_excess_return(
        self, validation_df: pd.DataFrame, benchmark_returns: pd.Series, horizon: int = 5
    ) -> Dict:
        col = f"ret_{horizon}d"
        if col not in validation_df.columns:
            return {"excess_return": 0.0, "count": 0}
        valid_df = validation_df.dropna(subset=[col]).copy()
        if valid_df.empty:
            return {"excess_return": 0.0, "count": 0}
        valid_df["benchmark_ret"] = valid_df["date"].map(
            lambda d: benchmark_returns.get(d, 0.0))
        valid_df["excess"] = valid_df[col] - valid_df["benchmark_ret"]
        return {
            "excess_return": valid_df["excess"].mean(),
            "excess_median": valid_df["excess"].median(),
            "excess_positive_rate": (valid_df["excess"] > 0).mean(),
            "count": len(valid_df)
        }

    def generate_validation_report(
        self, validation_df: pd.DataFrame, output_dir: str = "output/validation"
    ) -> Dict:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        report = {"summary": {}, "by_horizon": {}, "by_rank": {}, "by_industry": {}}
        for horizon in [5, 10, 20]:
            stats = self.calculate_hit_rate(validation_df, horizon)
            report["by_horizon"][f"{horizon}d"] = stats
        if "rank" in validation_df.columns:
            for rank_group in [(1, 10), (11, 20), (21, 30)]:
                mask = (validation_df["rank"] >= rank_group[0]) & \
                       (validation_df["rank"] <= rank_group[1])
                group_df = validation_df[mask]
                report["by_rank"][f"rank_{rank_group[0]}_{rank_group[1]}"] = \
                    self.calculate_hit_rate(group_df, 5)
        if "industry" in validation_df.columns:
            for industry in validation_df["industry"].unique():
                if pd.notna(industry):
                    group_df = validation_df[validation_df["industry"] == industry]
                    report["by_industry"][industry] = \
                        self.calculate_hit_rate(group_df, 5)
        report_file = output_path / "validation_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        validation_df.to_csv(output_path / "validation_details.csv", index=False)
        logger.info("验证报告已保存到: %s", output_path)
        return report