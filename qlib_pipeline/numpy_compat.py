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
    rtol = float(rtol)
    atol = float(atol)
    # Handle empty arrays (Qlib rolling std may return empty)
    a = _np.asarray(a)
    b = _np.asarray(b)
    if a.shape != b.shape:
        # Broadcasting mismatch -> return all False (safe default for Qlib)
        return _np.zeros(max(a.size, b.size), dtype=bool).reshape(
            _np.broadcast_shapes(a.shape, b.shape)
        )
    return _original_isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

np.isclose = _patched_isclose
