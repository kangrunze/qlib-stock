# -*- coding: utf-8 -*-
"""
[DEPRECATED] 自研回测引擎 — 已被 Qlib SimulatorExecutor + PortAnaRecord 替代。

请使用 Qlib Pipeline 统一入口:
    python run.py full      # 一键跑通
    python run.py backtest --rid <id>  # 回测分析

本模块保留用于向后兼容，不再维护。
"""

import warnings
warnings.warn(
    "backtest/backtest_engine.py 已被 Qlib SimulatorExecutor + PortAnaRecord 替代。"
    "请使用: python run.py full 或 python run.py backtest --rid <id>",
    DeprecationWarning, stacklevel=2
)