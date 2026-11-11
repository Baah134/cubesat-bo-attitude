"""Expert baseline with zero optimization budget: Wie-Lu rate-limited eigenaxis
quaternion feedback (stage "W"), parameters set by textbook design rules.

    w_max = 0.8 * 0.9 * h_max / J_max   slew rate whose body momentum stays inside
                                         the 90% wheel-momentum spec with 20% margin
    wn    = sqrt(tau_max / (J_max * theta_s))   same torque-limit bandwidth rule
                                         as the analytic PID baseline
    zeta  = 0.9                         well damped
    ki    = 0.1                         integral corner a decade below wn
No search: this is what a GNC engineer would write down from the spacecraft data
sheet before any tuning.
"""
import numpy as np

from cubesat.params import DEG, SLEW


def wie_rule_params(problem):
    sc = problem.sc
    Jmax = float(max(sc.J))
    w_max = 0.8 * 0.9 * sc.h_max / Jmax
    wn = np.sqrt(sc.tau_max / (Jmax * SLEW.angle_deg * DEG))
    return np.array([wn, 0.9, w_max, 0.1])


def optimize(problem, budget=0, seed=0):
    if problem.stage != "W":
        raise ValueError("wie_rule runs on stage W (expert control law)")
    g = np.clip(wie_rule_params(problem), 10 ** problem.log_lo, 10 ** problem.log_hi)
    rec = problem.evaluate(problem.to_unit(g))
    rec["iter"] = 0
    return [rec]
