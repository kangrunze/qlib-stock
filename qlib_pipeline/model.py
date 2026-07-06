# -*- coding: utf-8 -*-
"""
Qlib Pipeline Model 配置

提供 LGBModel 配置的兜底创建函数。
正常运行时，所有超参数来自 workflow_config.yaml 的 qlib_lgb 段；
本函数仅在配置文件缺失 qlib_lgb 段时作为兜底。
"""


def create_rank_model(loss: str = "mse", **overrides) -> dict:
    """创建 LightGBM 模型配置字典（兜底，不含完整超参）。

    正常路径：workflow_config.yaml → qlib_lgb.kwargs → build_task()
    兜底路径：配置文件缺失 qlib_lgb 段时调用本函数
    """
    model_cfg = {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": loss,
            "num_threads": 4,
        },
    }
    model_cfg["kwargs"].update(overrides)
    return model_cfg