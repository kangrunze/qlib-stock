#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 features 目录重建 instruments 文件
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
