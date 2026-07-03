#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
详细诊断脚本：定位 cannot reshape array 错误的根源
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# 必须先导入 numpy_compat
from qlib_pipeline.numpy_compat import *  # noqa

import qlib
from qlib.data import D
import pandas as pd
import numpy as np
import traceback

def test_single_stock_feature(stock, feature_expr):
    """测试单只股票的单个特征"""
    try:
        df = D.features(
            instruments=[stock],
            fields=[feature_expr],
            start_time="2020-01-01",
            end_time="2024-12-31"
        )
        return True, None
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 70)
    print("详细诊断：定位 cannot reshape array 错误")
    print("=" * 70)
    
    # 初始化 Qlib
    print("\n[步骤 1] 初始化 Qlib...")
    try:
        qlib.init(provider_uri="D:/trae/qlib_bin", region="cn")
        print("✓ Qlib 初始化成功")
    except Exception as e:
        print(f"✗ Qlib 初始化失败: {e}")
        traceback.print_exc()
        return False
    
    # 获取股票列表
    print("\n[步骤 2] 获取股票列表...")
    try:
        instruments = D.instruments(market="all")
        print(f"✓ 获取到 {len(instruments)} 只股票")
    except Exception as e:
        print(f"✗ 获取股票列表失败: {e}")
        traceback.print_exc()
        return False
    
    # 测试基础特征
    print("\n[步骤 3] 测试基础特征 ($close)...")
    test_stocks = instruments[:5]
    for stock in test_stocks:
        success, error = test_single_stock_feature(stock, "$close")
        if success:
            print(f"  ✓ {stock}")
        else:
            print(f"  ✗ {stock}: {error}")
            return False
    
    # 测试 Alpha158 特征（逐个测试）
    print("\n[步骤 4] 测试 Alpha158 特征（逐个测试前 3 只股票）...")
    
    # Alpha158 的部分特征表达式
    alpha158_features = [
        "$close",
        "$open",
        "$high",
        "$low",
        "$volume",
        "($close-$open)/$open",  # KMID
        "($high-$low)/$open",    # KLEN
        "($close-Ref($close,1))/Ref($close,1)",  # ROC
        "Mean($close, 3)",      # MA3
        "Mean($close, 5)",      # MA5
    ]
    
    for i, stock in enumerate(instruments[:3]):
        print(f"\n  测试股票 {i+1}/3: {stock}")
        for feat in alpha158_features:
            success, error = test_single_stock_feature(stock, feat)
            if success:
                print(f"    ✓ {feat}")
            else:
                print(f"    ✗ {feat}: {error}")
                print(f"      完整错误: {error}")
                return False
    
    # 测试完整 Alpha158 handler
    print("\n[步骤 5] 测试完整 Alpha158 handler...")
    try:
        from qlib.contrib.data.handler import Alpha158
        
        handler = Alpha158(
            instruments=instruments[:10],  # 只用前 10 只股票测试
            start_time="2020-01-01",
            end_time="2024-12-31",
            fit_start_time="2020-01-01",
            fit_end_time="2023-12-31"
        )
        
        data = handler.fetch()
        print(f"✓ Alpha158 handler 成功，数据形状: {data.shape}")
        
    except Exception as e:
        print(f"✗ Alpha158 handler 失败: {e}")
        traceback.print_exc()
        return False
    
    # 测试完整数据集
    print("\n[步骤 6] 测试完整数据集创建...")
    try:
        from qlib.data.dataset import DatasetH
        
        dataset = DatasetH(
            handler=handler,
            segments={
                "train": ("2020-01-01", "2023-12-31"),
                "valid": ("2024-01-01", "2024-06-30"),
                "test": ("2024-07-01", "2024-12-31"),
            }
        )
        
        train_data = dataset.prepare("train")
        valid_data = dataset.prepare("valid")
        test_data = dataset.prepare("test")
        
        print(f"✓ 数据集创建成功")
        print(f"  训练集: {train_data.shape}")
        print(f"  验证集: {valid_data.shape}")
        print(f"  测试集: {test_data.shape}")
        
    except Exception as e:
        print(f"✗ 数据集创建失败: {e}")
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("✓ 所有诊断测试通过！")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
