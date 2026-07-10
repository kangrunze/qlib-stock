# -*- coding: utf-8 -*-
"""
A股分板块涨跌停 Exchange — a_stock_exchange.py

Qlib 原生 Exchange 的 limit_threshold 是单一标量，无法按板块区分
A股不同板块的涨跌停幅度：
  - 主板（sh60/sz00/sz30 非300）: ±10%
  - 创业板（sz300/sz301）: ±20%
  - 科创板（sh688/sh689）: ±20%
  - 北交所（bj）: ±30%
  - ST 股: ±5%

本模块提供 PerStockLimitExchange 子类，重写 _update_limit，
按股票代码前缀分组应用不同阈值，生成 limit_buy/limit_sell 列。
下游 check_stock_limit 只读这两列，无需其他改动。

⚠ 已知限制 — ST 股 ±5% 阈值未实现:
    ST 股的涨跌停限制为 ±5%，但回测引擎的 quote_df 中不包含
    PIT（Point-in-Time）ST 状态标记，无法在 _update_limit 中
    动态判断某只股票在某天是否处于 ST 状态。
    当前 ST 股被按主板 ±10% 处理，回测中会允许 ST 股在 ±5%~±10%
    区间成交，可能轻微高估策略对 ST 股的可交易性。
    缓解措施：stock_universe.exclude_st=true 在训练/回测阶段已
    过滤掉 ST 股（见 train.py:_get_st_stocks），因此实际回测池中
    基本不含 ST 股，此限制的影响极小。
"""

import logging

import numpy as np
import pandas as pd

from qlib.backtest.exchange import Exchange

logger = logging.getLogger(__name__)

# 板块涨跌停阈值（留 0.1% 容差避免浮点精度误拒）
# 注意：ST 股 ±5% 未实现（见文件头已知限制说明）
_BOARD_LIMITS = {
    "main": 0.099,    # 主板 ±10%（ST 股也归入此类，见已知限制）
    "chinext": 0.199, # 创业板 ±20%
    "star": 0.199,    # 科创板 ±20%
    "bj": 0.299,      # 北交所 ±30%
}


def _classify_board(instrument: str) -> str:
    """根据 Qlib 股票代码前缀判断板块。

    Args:
        instrument: Qlib 格式股票代码，如 "sh600000"、"sz300001"

    Returns:
        板块标识: "main" | "chinext" | "star" | "bj"
    """
    code = str(instrument).lower()
    # 科创板: sh688/sh689
    if code.startswith(("sh688", "sh689")):
        return "star"
    # 创业板: sz300/sz301
    if code.startswith(("sz300", "sz301")):
        return "chinext"
    # 北交所: bj
    if code.startswith("bj"):
        return "bj"
    # 主板: 其余
    return "main"


class PerStockLimitExchange(Exchange):
    """按 A 股板块区分涨跌停阈值的 Exchange。

    继承 Qlib 原生 Exchange，仅重写 _update_limit 方法。
    其余所有逻辑（成交价、手续费、成交量限制等）保持不变。

    使用方式：
      通过 monkey-patch qlib.backtest.get_exchange 注入，
      见 run.py 中的 _a_stock_exchange_context()。
    """

    def _update_limit(self, limit_threshold) -> None:
        """按股票代码前缀分组应用不同涨跌停阈值。

        参数 limit_threshold 被忽略（由板块规则决定阈值），
        保留参数签名以兼容父类接口。
        """
        suspended = self.quote_df["$close"].isna()

        # 获取股票代码（MultiIndex 第一级 instrument）
        idx = self.quote_df.index
        if isinstance(idx, pd.MultiIndex):
            instruments = idx.get_level_values(0)
        else:
            instruments = idx

        # 按板块分类，生成逐股票阈值
        boards = np.array([_classify_board(s) for s in instruments])
        thresholds = np.full(len(self.quote_df), _BOARD_LIMITS["main"], dtype=float)

        for board_name, limit_val in _BOARD_LIMITS.items():
            mask = boards == board_name
            if mask.any():
                thresholds[mask] = limit_val

        # 按个股阈值判断涨跌停
        changes = self.quote_df["$change"]
        self.quote_df["limit_buy"] = changes.ge(thresholds) | suspended
        self.quote_df["limit_sell"] = changes.le(-thresholds) | suspended

        n_star = (boards == "star").sum()
        n_chinext = (boards == "chinext").sum()
        n_bj = (boards == "bj").sum()
        n_main = (boards == "main").sum()
        logger.debug(
            "PerStockLimitExchange: 主板=%d, 创业板=%d, 科创板=%d, 北交所=%d, "
            "涨跌停阈值分别=%.1f%%/%.1f%%/%.1f%%/%.1f%%",
            n_main, n_chinext, n_star, n_bj,
            _BOARD_LIMITS["main"] * 100, _BOARD_LIMITS["chinext"] * 100,
            _BOARD_LIMITS["star"] * 100, _BOARD_LIMITS["bj"] * 100,
        )
