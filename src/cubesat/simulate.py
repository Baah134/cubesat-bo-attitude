"""Scenario runners. A run = one scenario, one parameter set, one noise seed.

Noise and random disturbance sequences are generated here with numpy's
PCG64 generator from an explicit seed, then passed into the (optionally
JIT-compiled) kernels, so results are identical with or without numba.
"""
from dataclasses import dataclass

import numpy as np

from . import single_axis
from .metrics import SETTLE_BAND_DEG, overshoot_pct, rms_after, settling_time
from .params import DEG, EnvParams, Scenario, SimParams, SpacecraftParams


@dataclass
class RunResult:
    t: np.ndarray
    err: np.ndarray          # true pointing error (rad); Stage A: theta - ref
    rate: np.ndarray         # body rate (rad/s); Stage B: (n,3)
    h: np.ndarray            # wheel momentum (N m s); Stage B: (n,3)
    tau: np.ndarray          # applied torque (N m); Stage B: (n,3)
    u_cmd: np.ndarray        # commanded torque at controller rate
    sat: np.ndarray          # saturation flag at controller rate
    h_clamped: np.ndarray    # momentum-limit flag at integration rate
    energy: float
    diverged: bool
    horizon: float
    travel: np.ndarray = None  # angle travelled along slew axis (rad), for overshoot


def _noise(seed, n_ctrl, n_axes, env: EnvParams):
    rng = np.random.default_rng(seed)
    att = rng.normal(0.0, env.att_noise_std, size=(n_ctrl, n_axes))
    gyro = rng.normal(0.0, env.gyro_noise_std, size=(n_ctrl, n_axes))
    drand = rng.normal(0.0, env.dist_rand_std, size=(n_ctrl, n_axes)) * env.dist_scale
    phase = rng.uniform(0.0, 2 * np.pi, size=n_axes)
    return att, gyro, drand, phase


def run_single_axis(gains, scenario: Scenario, sc: SpacecraftParams = SpacecraftParams(),
                    env: EnvParams = EnvParams(), sim: SimParams = SimParams(),
                    seed: int = 0, i_clamp_frac: float = 0.25, noise: bool = True) -> RunResult:
    """Stage A run about body x. gains = (Kp, Ki, Kd)."""
    Kp, Ki, Kd = (float(g) for g in gains)
    n_ctrl = int(round(scenario.horizon / sim.ctrl_dt))
    att, gyro, drand, phase = _noise(seed, n_ctrl, 1, env)
    if not noise:
        att[:] = 0.0
        gyro[:] = 0.0
        drand[:] = 0.0
    ang = scenario.angle_deg * DEG
    if scenario.kind == "slew":
        theta0, ref = 0.0, ang
    else:
        theta0, ref = ang, 0.0
    ds = env.dist_scale if noise else 0.0
    gb = env.gyro_bias[0] if noise else 0.0
    out = single_axis.simulate_kernel(
        theta0, ref, Kp, Ki, Kd, i_clamp_frac * sc.tau_max,
        sc.J[0], sc.J_wheel, sc.T_m, sc.tau_max, sc.h_max,
        env.dist_bias[0] * ds, env.dist_sin_amp[0] * ds, env.orbit_period, phase[0],
        drand[:, 0].copy(), att[:, 0].copy(), gyro[:, 0].copy(), gb,
        scenario.horizon, sim.dt, sim.ctrl_dt)
    t, th, om, h, tau, u, sat, hcl, E, div = out
    return RunResult(t, th - ref, om, h, tau, u, sat, hcl, float(E), bool(div), scenario.horizon,
                     travel=th - theta0)


def scenario_metrics(r: RunResult, scenario: Scenario, sc: SpacecraftParams) -> dict:
    """Metrics for one run. Stage B passes err as the principal error angle (>=0)."""
    band = SETTLE_BAND_DEG * DEG
    err = r.err
    out = {
        "diverged": bool(r.diverged or not np.all(np.isfinite(err))),
        "t_settle": settling_time(r.t, err, band, r.horizon) if not r.diverged else r.horizon,
        "tau_peak_frac": float(np.max(np.abs(r.u_cmd)) / sc.tau_max) if r.u_cmd.size else 0.0,
        "sat_frac": float(np.mean(r.sat)) if r.sat.size else 0.0,
        "h_peak_frac": float(np.max(np.abs(r.h)) / sc.h_max),
        "h_violation": bool(np.any(r.h_clamped)),
        "energy": r.energy,
    }
    if scenario.kind == "slew":
        target = scenario.angle_deg * DEG
        # r.travel: rotation angle achieved along the commanded slew axis
        out["overshoot"] = overshoot_pct(r.travel, target)
        out["e_rms"] = rms_after(r.t, err, r.horizon * 0.5)
    else:
        out["e_rms"] = rms_after(r.t, err, scenario.transient)
        out["overshoot"] = 0.0
    return out
