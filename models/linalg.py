"""Float64 BLAS wrapper with explicit finite-output checking.

Direct SciPy BLAS avoids NumPy matmul's spurious floating-point status warnings
on some Apple Accelerate builds (numpy/numpy#28687). No warnings are suppressed.
"""
import numpy as np
from scipy.linalg.blas import dgemm


def matmul(a, b):
    result = dgemm(1., np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    if not np.isfinite(result).all():
        raise FloatingPointError('Non-finite matrix product')
    return result
