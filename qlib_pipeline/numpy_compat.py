# -*- coding: utf-8 -*-
"""
Numpy 2.x compatibility patch for Qlib 0.9.7.

Qlib 0.9.7 uses np.isclose with atol parameter that may cause TypeError
in numpy 2.x when the rolling std returns integer-like dtypes or empty arrays.
This patch must be imported BEFORE importing qlib.

注意：如果项目已锁定 numpy<2.0，本补丁理论上已无必要。当前保留作为安全兜底，
所有异常分支均已加 logger.warning 以便追踪是否仍有触发。
"""
import logging
import numpy as np

logger = logging.getLogger("numpy_compat")

_original_isclose = np.isclose

def _patched_isclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False):
    """Patched np.isclose that handles edge cases with numpy 2.x."""
    import numpy as _np
    
    # Convert to arrays first
    a_arr = _np.asarray(a)
    b_arr = _np.asarray(b)
    
    # Check for empty arrays BEFORE calling original function
    if a_arr.size == 0 or b_arr.size == 0:
        try:
            out_shape = _np.broadcast_shapes(a_arr.shape, b_arr.shape)
            logger.debug("numpy_compat: 空数组兜底，broadcast_shapes=%s", out_shape)
            return _np.zeros(out_shape, dtype=bool)
        except ValueError:
            logger.warning("numpy_compat: 空数组 broadcast 失败，返回空 bool 数组")
            return _np.zeros(0, dtype=bool)
    
    # For non-empty arrays, use the original function
    try:
        return _original_isclose(a_arr, b_arr, rtol=rtol, atol=atol, equal_nan=equal_nan)
    except (ValueError, TypeError) as e:
        logger.warning("numpy_compat: np.isclose 回退到零数组（原因: %s），请检查 numpy 版本兼容性", e)
        try:
            out_shape = _np.broadcast_shapes(a_arr.shape, b_arr.shape)
            return _np.zeros(out_shape, dtype=bool)
        except ValueError:
            logger.warning("numpy_compat: broadcast 回退也失败，返回空 bool 数组")
            return _np.zeros(0, dtype=bool)

np.isclose = _patched_isclose
