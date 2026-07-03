"""Debug: comprehensive data integrity check"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from qlib_pipeline.numpy_compat import *  # noqa

import qlib
from qlib.data import D
import pandas as pd

print("=" * 60)
print("Step 1: Initialize Qlib")
print("=" * 60)
qlib.init(provider_uri="D:/trae/qlib_bin", region="cn")

print("\n" + "=" * 60)
print("Step 2: Check calendar")
print("=" * 60)
try:
    cal = D.calendar(start_time="2020-01-01", end_time="2026-06-25")
    print(f"✓ Calendar loaded: {len(cal)} trading days")
    print(f"  First: {cal[0]}, Last: {cal[-1]}")
except Exception as e:
    print(f"✗ Calendar error: {e}")

print("\n" + "=" * 60)
print("Step 3: Check instruments file format")
print("=" * 60)
inst_path = r"D:\trae\qlib_bin\instruments\all.txt"
with open(inst_path, "r") as f:
    lines = f.readlines()[:5]
print("First 5 lines of all.txt:")
for i, line in enumerate(lines):
    print(f"  {i}: {repr(line)}")

print("\n" + "=" * 60)
print("Step 4: Load instruments via Qlib API")
print("=" * 60)
try:
    instruments = D.instruments(market="all")
    print(f"✓ Instruments loaded: {len(instruments)} stocks")
    print(f"  Sample: {instruments[:3]}")
except Exception as e:
    print(f"✗ Instruments error: {e}")

print("\n" + "=" * 60)
print("Step 5: Try loading features for single stock")
print("=" * 60)
# Get a stock from features directory
features_dir = r"D:\trae\qlib_bin\features"
sample_stock = os.listdir(features_dir)[0]
print(f"Testing with stock: {sample_stock}")

try:
    df = D.features(
        instruments=[sample_stock],
        fields=["$close", "$volume"],
        start_time="2025-01-01",
        end_time="2025-01-31"
    )
    print(f"✓ Features loaded: shape={df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Index type: {type(df.index)}")
    if not df.empty:
        print(f"  First few rows:\n{df.head()}")
except Exception as e:
    print(f"✗ Features error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Step 6: Try loading with Alpha158 handler")
print("=" * 60)
try:
    from qlib.contrib.data.handler import Alpha158
    
    handler = Alpha158(
        instruments="all",
        start_time="2020-01-01",
        end_time="2026-06-25",
        fit_start_time="2020-01-01",
        fit_end_time="2023-06-30"
    )
    print(f"✓ Alpha158 handler created")
    print(f"  Data shape: {handler.data.shape if hasattr(handler, 'data') else 'N/A'}")
except Exception as e:
    print(f"✗ Alpha158 handler error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Done")
print("=" * 60)
