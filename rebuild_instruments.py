#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
[仅供调试使用 — 不要在正式流程中调用]

从 features 目录重建 instruments 文件。

⚠ 已知问题（生存者偏差 + 上市日期失真）:
    本脚本把 features 目录下当前还存在的股票统一标记
    2020-01-01 ~ 2026-06-25，存在两个严重问题：
    1. 生存者偏差: 已退市且 feature 文件已被清理的股票永远不会出现，
       长周期回测会系统性高估收益（尤其熊市尾部的退市股缺失）。
    2. 上市日期失真: 对 2020 年后上市的股票，start_date 仍写 2020-01-01，
       导致回测引擎尝试加载不存在的早期数据。
    正式流程应从 all.txt 的真实 start/end 读取，或通过
    `python run.py data --convert` 重建（该流程会正确记录每只股票的真实上市区间）。

仅当 instruments 文件损坏（如混入二进制数据）且需要快速恢复到
"能跑通"状态时，才可临时使用本脚本。
"""
import os
from pathlib import Path

def rebuild_instruments():
    provider_uri = os.environ.get("QLIB_PROVIDER_URI")
    if not provider_uri:
        raise ValueError("未设置 QLIB_PROVIDER_URI 环境变量")
    qlib_root = Path(provider_uri)
    features_dir = qlib_root / "features"
    instruments_dir = qlib_root / "instruments"
    
    # Qlib instruments 文件格式: symbol\tstart_date\tend_date
    start_date = "2020-01-01"
    end_date = "2026-06-25"
    
    # 获取所有股票代码（从 features 目录）
    stock_codes = sorted(os.listdir(features_dir))
    print(f"从 features 目录读取到 {len(stock_codes)} 个股票")
    
    # 生成 all.txt (3列格式)
    all_txt_path = instruments_dir / "all.txt"
    with open(all_txt_path, "w", encoding="utf-8") as f:
        for code in stock_codes:
            f.write(f"{code}\t{start_date}\t{end_date}\n")
    print(f"✓ 已生成 {all_txt_path}")
    
    # 按交易所分类
    sh_stocks = [s for s in stock_codes if s.startswith("sh")]
    sz_stocks = [s for s in stock_codes if s.startswith("sz")]
    bj_stocks = [s for s in stock_codes if s.startswith("bj")]
    
    print(f"  上交所: {len(sh_stocks)} 只")
    print(f"  深交所: {len(sz_stocks)} 只")
    print(f"  北交所: {len(bj_stocks)} 只")
    
    # 生成 sh.txt
    sh_txt_path = instruments_dir / "sh.txt"
    with open(sh_txt_path, "w", encoding="utf-8") as f:
        for code in sh_stocks:
            f.write(f"{code}\t{start_date}\t{end_date}\n")
    print(f"✓ 已生成 {sh_txt_path}")
    
    # 生成 sz.txt
    sz_txt_path = instruments_dir / "sz.txt"
    with open(sz_txt_path, "w", encoding="utf-8") as f:
        for code in sz_stocks:
            f.write(f"{code}\t{start_date}\t{end_date}\n")
    print(f"✓ 已生成 {sz_txt_path}")
    
    # 生成 bj.txt
    bj_txt_path = instruments_dir / "bj.txt"
    with open(bj_txt_path, "w", encoding="utf-8") as f:
        for code in bj_stocks:
            f.write(f"{code}\t{start_date}\t{end_date}\n")
    print(f"✓ 已生成 {bj_txt_path}")
    
    print("\n所有 instruments 文件已重建完成！")

if __name__ == "__main__":
    rebuild_instruments()
