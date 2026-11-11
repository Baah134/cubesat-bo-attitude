"""Analytic (textbook) baseline: zero optimization budget.

Per axis with inertia J and slew error amplitude theta_s:
    Kp = tau_max / theta_s            (PD output just reaches the torque limit at slew start)
    wn = sqrt(Kp / J),  zeta = 0.9    (well damped, to limit overshoot under saturation)
    Kd = 2 * zeta * wn * J
    Ki = Kp * wn / 10                 (integral corner a decade below wn)
This is a bandwidth rule chosen from the torque limit. It stands in for a
"textbook" design; a MATLAB pidtune baseline is produced separately
(scripts/pidtune_baseline.m) because MATLAB is available on this machine.
"""
import numpy as np

from cubesat.params import DEG, SLEW

ZETA = 0.9


def axis_gains(J, theta_s, tau_max, zeta=ZETA):
    Kp = tau_max / theta_s
    wn = np.sqrt(Kp / J)
    return Kp, Kp * wn / 10.0, 2 * zeta * wn * J


def analytic_gains(problem):
    sc = problem.sc
    th = SLEW.angle_deg * DEG
    if problem.stage == "A":
        return np.array(axis_gains(sc.J[0], th, sc.tau_max))
    # Stage B: slew about [1,1,1]/sqrt(3) -> per-axis error component th/sqrt(3)
    ax = np.abs(np.asarray(SLEW.axis))
    gx = axis_gains(sc.J[0], th * ax[0], sc.tau_max)
    gz = axis_gains(sc.J[2], th * ax[2], sc.tau_max)
    return np.array([*gx, *gz])


def optimize(problem, budget=0, seed=0):
    g = np.clip(analytic_gains(problem), 10 ** problem.log_lo, 10 ** problem.log_hi)
    rec = problem.evaluate(problem.to_unit(g))
    rec["iter"] = 0
    return [rec]
