# -*- coding: utf-8 -*-
"""推荐跟踪器 - 保存和跟踪每日推荐"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class RecommendationTracker:
    """推荐跟踪器 - 管理每日股票推荐记录"""
    
    def __init__(self, output_dir: str = "output/recommendations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.recommendations_file = self.output_dir / "recommendations.jsonl"
        self.daily_file = self.output_dir / "daily_picks"
        self.daily_file.mkdir(parents=True, exist_ok=True)
        
    def save_recommendation(
        self,
        date: str,
        stock_code: str,
        score: float,
        rank: int,
        industry: str = "",
        reason: str = "",
        metadata: Optional[Dict] = None
    ):
        """保存单条推荐记录"""
        record = {
            "date": date,
            "stock_code": stock_code,
            "score": float(score),
            "rank": int(rank),
            "industry": industry,
            "reason": reason,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        with open(self.recommendations_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
        logger.info(f"保存推荐: {date} {stock_code} rank={rank} score={score:.4f}")
        
    def save_daily_picks(
        self,
        date: str,
        picks_df: pd.DataFrame,
        top_k: int = 30
    ):
        """保存每日推荐列表
        
        Args:
            date: 日期 YYYY-MM-DD
            picks_df: DataFrame with columns [stock_code, score, industry, ...]
            top_k: 推荐数量
        """
        picks_df = picks_df.head(top_k).copy()
        picks_df["date"] = date
        picks_df["rank"] = range(1, len(picks_df) + 1)
        
        # 保存 CSV
        csv_file = self.daily_file / f"picks_{date}.csv"
        picks_df.to_csv(csv_file, index=False, encoding="utf-8-sig")
        
        # 同时保存到 JSONL
        for _, row in picks_df.iterrows():
            self.save_recommendation(
                date=date,
                stock_code=row.get("stock_code", row.get("instrument", "")),
                score=row.get("score", 0.0),
                rank=row.get("rank", 0),
                industry=row.get("industry", ""),
                reason=row.get("reason", "AI Score")
            )
            
        logger.info(f"保存每日推荐: {date} Top-{top_k} -> {csv_file}")
        
    def load_recommendations(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        stock_code: Optional[str] = None
    ) -> pd.DataFrame:
        """加载推荐记录"""
        if not self.recommendations_file.exists():
            return pd.DataFrame()
            
        records = []
        with open(self.recommendations_file, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                records.append(record)
                
        df = pd.DataFrame(records)
        
        # 过滤日期
        if start_date:
            df = df[df["date"] >= start_date]
        if end_date:
            df = df[df["date"] <= end_date]
            
        # 过滤股票
        if stock_code:
            df = df[df["stock_code"] == stock_code]
            
        return df.sort_values(["date", "rank"])
        
    def get_latest_recommendations(self, date: Optional[str] = None) -> pd.DataFrame:
        """获取最新一天的推荐"""
        df = self.load_recommendations()
        if df.empty:
            return df
            
        if date is None:
            date = df["date"].max()
            
        return df[df["date"] == date].sort_values("rank")
        
    def get_recommendation_stats(self, days: int = 30) -> Dict:
        """获取推荐统计信息"""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        df = self.load_recommendations(start_date=start_date, end_date=end_date)
        
        if df.empty:
            return {"total": 0, "days": 0, "stocks": 0}
            
        return {
            "total": len(df),
            "days": df["date"].nunique(),
            "stocks": df["stock_code"].nunique(),
            "avg_score": df["score"].mean(),
            "date_range": f"{df['date'].min()} ~ {df['date'].max()}"
        }
