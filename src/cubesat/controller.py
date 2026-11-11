"""Quaternion-feedback PID with anti-windup and per-axis saturation.

Control law (per axis i), producing a *body* torque command u:

    u_i = -(Kp_i * e_i + Kd_i * w_i + Ki_i * I_i),   |u_i| <= tau_max

where e is the attitude error (rad, small-angle vector part of the error
quaternion times 2, or the angle itself in Stage A), w the measured body rate
and I the error integral. The wheels apply -u to themselves (reaction).

Anti-windup (both active):
  1. clamping:   |Ki_i * I_i| <= i_clamp   (torque units)
  2. conditional integration: skip the integrator update when the output is
     saturated and integrating would push it further into saturation.
"""
from ._jit import njit


@njit
def pid_axis(e, rate, I, Kp, Ki, Kd, ctrl_dt, i_clamp, tau_max):
    """One PID update for one axis. Returns (u_sat, I_new, saturated)."""
    I_try = I + e * ctrl_dt
    # clamp integrator contribution
    if Ki > 0.0:
        lim = i_clamp / Ki
        if I_try > lim:
            I_try = lim
        elif I_try < -lim:
            I_try = -lim
    u_try = -(Kp * e + Kd * rate + Ki * I_try)
    I_new = I_try
    if u_try > tau_max or u_try < -tau_max:
        # integrating pushes output further out if -Ki*e has the sign of u_try
        if (u_try > 0.0 and -Ki * e > 0.0) or (u_try < 0.0 and -Ki * e < 0.0):
            I_new = I
    u = -(Kp * e + Kd * rate + Ki * I_new)
    sat = False
    if u > tau_max:
        u = tau_max
        sat = True
    elif u < -tau_max:
        u = -tau_max
        sat = True
    return u, I_new, sat
