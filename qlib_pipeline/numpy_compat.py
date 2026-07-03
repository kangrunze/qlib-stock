# -*- coding: utf-8 -*-
"""
Numpy 2.x compatibility patch for Qlib 0.9.7.

Qlib 0.9.7 uses np.isclose with atol parameter that may cause TypeError
in numpy 2.x when the rolling std returns integer-like dtypes or empty arrays.
This patch must be imported BEFORE importing qlib.
"""
import numpy as np

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
            return _np.zeros(out_shape, dtype=bool)
        except ValueError:
            # If broadcast fails, return empty array
            return _np.zeros(0, dtype=bool)
    
    # For non-empty arrays, use the original function
    try:
        return _original_isclose(a_arr, b_arr, rtol=rtol, atol=atol, equal_nan=equal_nan)
    except (ValueError, TypeError) as e:
        # Fallback for any broadcasting or type issues
        try:
            out_shape = _np.broadcast_shapes(a_arr.shape, b_arr.shape)
            return _np.zeros(out_shape, dtype=bool)
        except ValueError:
            # If broadcast fails, return empty array
            return _np.zeros(0, dtype=bool)

np.isclose = _patched_isclose
