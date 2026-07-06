# -*- coding: utf-8 -*-
# research/ — 研究分析模块
#
# Phase 1 验证方法论基础设施:
#   - significance: 统计显著性检验（Newey-West + 多重检验校正）
#   - risk_model: 风险模型（Barra 风格因子暴露）
#   - attribution: 收益归因分析
#   - experiment_tracker: 探索/确认两阶段实验追踪
#
# Phase 2 中长周期策略:
#   - portfolio_constructor: 行业中性化策略（继承 Qlib TopkDropoutStrategy）
#   - capacity: 策略容量分析
#
# Phase 4 模型能力恢复:
#   - explain: SHAP 可解释性分析

from research.significance import (
    newey_west_se,
    newey_west_test,
    multiple_testing_correction,
    ic_significance_report,
)

from research.risk_model import (
    calculate_style_exposures,
    check_style_constraints,
    get_industry_exposures,
    check_industry_constraints,
    get_style_factor_names,
)

from research.attribution import (
    decompose_excess_return,
    calculate_factor_returns,
    attribution_report,
)

from research.portfolio_constructor import (
    IndustryConstrainedTopkStrategy,
    create_industry_map_from_csv,
    create_industry_map_from_akshare,
)

from research.capacity import (
    check_single_stock_capacity,
    check_portfolio_capacity,
    get_average_volume_from_data,
)

from research.experiment_tracker import (
    ExperimentTracker,
)

from research.explain import (
    compute_shap_values,
    generate_shap_report,
    get_shap_feature_importance,
    explain_single_prediction,
)