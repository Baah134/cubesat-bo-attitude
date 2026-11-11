"""Optional numba JIT. Falls back to plain Python if numba is unavailable.

Results must not depend on whether numba is present; tests run both paths
when numba is installed (see tests/test_determinism.py).
"""
import os

try:  # pragma: no cover - import guard
    if os.environ.get("CUBESAT_NO_JIT") == "1":
        raise ImportError
    from numba import njit as _njit

    def njit(fn):
        return _njit(cache=True, fastmath=False)(fn)

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    def njit(fn):
        return fn

    HAVE_NUMBA = False
