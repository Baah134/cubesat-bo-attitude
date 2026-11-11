"""Performance metrics computed from simulation outputs.

All metrics use the TRUE attitude (not the noisy measurement).
"""
import numpy as np

SETTLE_BAND_DEG = 0.1


def settling_time(t, err, band, horizon):
    """Time after which |err| stays <= band. Returns horizon if it never settles
    (or is still outside the band at the final sample)."""
    err = np.asarray(err)
    outside = np.nonzero(np.abs(err) > band)[0]
    if outside.size == 0:
        return 0.0
    last = outside[-1]
    if last >= len(t) - 1:
        return float(horizon)
    return float(t[last + 1])


def overshoot_pct(angle, target):
    """Peak travel beyond target, as % of |target|. Assumes motion from 0 to target."""
    angle = np.asarray(angle)
    if target == 0:
        raise ValueError("overshoot undefined for zero target")
    beyond = (angle - target) * np.sign(target)
    return float(max(0.0, beyond.max()) / abs(target) * 100.0)


def rms_after(t, err, t0):
    err = np.asarray(err)
    m = np.asarray(t) >= t0
    if not m.any():
        return float("nan")
    return float(np.sqrt(np.mean(err[m] ** 2)))
