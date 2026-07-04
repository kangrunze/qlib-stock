# -*- coding: utf-8 -*-
"""推荐仪表盘 - 生成推荐统计和可视化"""

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class RecommendationDashboard:
    """推荐仪表盘 - 生成推荐统计报告"""
    
    def __init__(self, tracker, validator=None):
        self.tracker = tracker
        self.validator = validator
        
    def generate_summary(
        self,
        days: int = 30,
        output_dir: str = "output/recommendations/dashboard"
    ) -> Dict:
        """生成推荐摘要"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 加载推荐记录
        recommendations = self.tracker.load_recommendations()
        if recommendations.empty:
            logger.warning("没有推荐记录")
            return {}
            
        # 基础统计
        stats = self.tracker.get_recommendation_stats(days)
        
        # 按日期统计
        daily_stats = recommendations.groupby("date").agg({
            "stock_code": "count",
            "score": "mean"
        }).rename(columns={"stock_code": "count", "score": "avg_score"})
        
        # 按行业统计
        if "industry" in recommendations.columns:
            industry_stats = recommendations.groupby("industry").agg({
                "stock_code": "count",
                "score": "mean"
            }).rename(columns={"stock_code": "count", "score": "avg_score"})
        else:
            industry_stats = pd.DataFrame()
            
        # 生成报告
        summary = {
            "overall": stats,
            "daily_stats": daily_stats.to_dict(),
            "industry_stats": industry_stats.to_dict() if not industry_stats.empty else {},
            "latest_date": recommendations["date"].max(),
            "total_recommendations": len(recommendations)
        }
        
        # 保存摘要
        import json
        summary_file = output_path / "dashboard_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
            
        # 保存 CSV
        daily_stats.to_csv(output_path / "daily_stats.csv")
        if not industry_stats.empty:
            industry_stats.to_csv(output_path / "industry_stats.csv")
            
        logger.info(f"推荐仪表盘已保存到: {output_path}")
        return summary
        
    def generate_html_report(
        self,
        output_file: str = "output/recommendations/dashboard.html"
    ):
        """生成 HTML 报告"""
        summary = self.generate_summary()
        if not summary:
            return
            
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>推荐仪表盘</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .stat-box {{ 
                    display: inline-block; 
                    padding: 20px; 
                    margin: 10px; 
                    background: #f5f5f5; 
                    border-radius: 5px; 
                }}
                .stat-value {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
                .stat-label {{ font-size: 14px; color: #666; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #2196F3; color: white; }}
            </style>
        </head>
        <body>
            <h1>推荐仪表盘</h1>
            
            <div class="stat-box">
                <div class="stat-value">{summary['overall'].get('total', 0)}</div>
                <div class="stat-label">总推荐数</div>
            </div>
            
            <div class="stat-box">
                <div class="stat-value">{summary['overall'].get('days', 0)}</div>
                <div class="stat-label">推荐天数</div>
            </div>
            
            <div class="stat-box">
                <div class="stat-value">{summary['overall'].get('stocks', 0)}</div>
                <div class="stat-label">推荐股票数</div>
            </div>
            
            <div class="stat-box">
                <div class="stat-value">{summary['overall'].get('avg_score', 0):.4f}</div>
                <div class="stat-label">平均得分</div>
            </div>
            
            <h2>每日统计</h2>
            <table>
                <tr><th>日期</th><th>推荐数量</th><th>平均得分</th></tr>
        """
        
        # 添加每日数据
        daily = summary.get("daily_stats", {})
        for date in sorted(daily.get("date", {}).keys(), reverse=True)[:10]:
            count = daily.get("count", {}).get(date, 0)
            avg_score = daily.get("avg_score", {}).get(date, 0)
            html += f"<tr><td>{date}</td><td>{count}</td><td>{avg_score:.4f}</td></tr>"
            
        html += """
            </table>
        </body>
        </html>
        """
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html)
            
        logger.info(f"HTML 报告已保存到: {output_file}")
