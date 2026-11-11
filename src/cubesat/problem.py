"""Map a normalized parameter vector x in [0,1]^d to metrics, objective and constraints.

Convention (same as the IECON framework): constraint values c_i <= 0 mean
feasible. The objective is slew settling time t_settle (minimize). For the
GP-based optimizer we also expose log-transformed outcomes (see `outcomes`).

Diverged / crashed runs never produce NaN or inf: metrics are clipped to
documented worst-case caps and every constraint is marked violated.
"""
import time
from dataclasses import dataclass, field

import numpy as np

from .params import DEG, HOLD, SLEW, EnvParams, SimParams, SpacecraftParams

# Worst-case caps used for diverged runs and to bound heavy tails.
CAP_OVERSHOOT = 200.0      # %
CAP_E_RMS_DEG = 10.0       # deg
CAP_FRAC = 1.0


@dataclass(frozen=True)
class Specs:
    max_overshoot: float = 5.0          # % of slew angle
    max_e_hold_rms_deg: float = 0.002   # deg RMS during hold (calibrated in M1 from 0.02; see DECISIONS.md)
    max_h_peak_frac: float = 0.9        # keep 10% momentum margin
    max_sat_frac: float = 0.2           # fraction of slew controller samples saturated

    @property
    def names(self):
        return ["overshoot", "e_hold_rms", "h_peak", "sat_frac"]


# Log-space gain bounds. Stage A: (Kp, Ki, Kd). Justified in docs/EXPLAINER.md.
BOUNDS_A = {
    "Kp": (1e-5, 1.0),     # N m / rad
    "Ki": (1e-8, 1e-2),    # N m / (rad s)
    "Kd": (1e-4, 1.0),     # N m s / rad
}


@dataclass
class Problem:
    stage: str = "A"
    specs: Specs = field(default_factory=Specs)
    sc: SpacecraftParams = field(default_factory=SpacecraftParams)
    env: EnvParams = field(default_factory=EnvParams)
    sim: SimParams = field(default_factory=SimParams)
    slew_seed: int = 1000      # fixed nominal noise seeds (same for every optimizer)
    hold_seed: int = 2000
    i_clamp_frac: float = 0.25

    def __post_init__(self):
        if self.stage == "A":
            self.names = list(BOUNDS_A)
            lo = np.array([BOUNDS_A[k][0] for k in self.names])
            hi = np.array([BOUNDS_A[k][1] for k in self.names])
        else:
            # stage "B": three-axis PID; stage "W": three-axis Wie-Lu expert law
            from .three_axis import BOUNDS_B, BOUNDS_W
            bd = BOUNDS_W if self.stage == "W" else BOUNDS_B
            self.names = list(bd)
            lo = np.array([bd[k][0] for k in self.names])
            hi = np.array([bd[k][1] for k in self.names])
        self.log_lo, self.log_hi = np.log10(lo), np.log10(hi)
        self.dim = len(self.names)
        self.n_constraints = 4

    # ----- parameter mapping -----
    def to_gains(self, x):
        x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
        return 10.0 ** (self.log_lo + x * (self.log_hi - self.log_lo))

    def to_unit(self, gains):
        g = np.log10(np.asarray(gains, dtype=float))
        return np.clip((g - self.log_lo) / (self.log_hi - self.log_lo), 0.0, 1.0)

    # ----- simulation -----
    def simulate(self, gains, sc=None, env=None, slew_seed=None, hold_seed=None, slew=None):
        """slew: slew scenario (default SLEW, 30 deg); LARGE_SLEW is used for the generalization test."""
        from .simulate import scenario_metrics
        sc = sc or self.sc
        slew = slew or SLEW
        env = env or self.env
        ss = self.slew_seed if slew_seed is None else slew_seed
        hs = self.hold_seed if hold_seed is None else hold_seed
        if self.stage == "A":
            from .simulate import run_single_axis
            rs = run_single_axis(gains, slew, sc, env, self.sim, ss, self.i_clamp_frac)
            rh = run_single_axis(gains, HOLD, sc, env, self.sim, hs, self.i_clamp_frac)
        else:
            from .three_axis import run_three_axis
            law = 1 if self.stage == "W" else 0
            rs = run_three_axis(gains, slew, sc, env, self.sim, ss, self.i_clamp_frac, law=law)
            rh = run_three_axis(gains, HOLD, sc, env, self.sim, hs, self.i_clamp_frac, law=law)
        ms = scenario_metrics(rs, slew, sc)
        mh = scenario_metrics(rh, HOLD, sc)
        return self._combine(ms, mh, slew.horizon), (rs, rh)

    def _combine(self, ms, mh, horizon=SLEW.horizon):
        diverged = ms["diverged"] or mh["diverged"]
        m = {
            "t_settle": float(ms["t_settle"]),
            "overshoot": float(ms["overshoot"]),
            "e_hold_rms_deg": float(mh["e_rms"] / DEG),
            "hold_t_settle": float(mh["t_settle"]),
            "h_peak_frac": float(max(ms["h_peak_frac"], mh["h_peak_frac"])),
            "h_violation": bool(ms["h_violation"] or mh["h_violation"]),
            "sat_frac": float(ms["sat_frac"]),
            "tau_peak_frac": float(max(ms["tau_peak_frac"], mh["tau_peak_frac"])),
            "energy": float(ms["energy"] + mh["energy"]),
            "diverged": bool(diverged),
        }
        if diverged or not np.isfinite(m["e_hold_rms_deg"]):
            m.update(t_settle=horizon, overshoot=CAP_OVERSHOOT, e_hold_rms_deg=CAP_E_RMS_DEG,
                     h_peak_frac=CAP_FRAC, sat_frac=CAP_FRAC, energy=float("nan"), diverged=True)
        m["overshoot"] = min(m["overshoot"], CAP_OVERSHOOT)
        m["e_hold_rms_deg"] = min(m["e_hold_rms_deg"], CAP_E_RMS_DEG)
        return m

    # ----- objective / constraints -----
    def constraints(self, m):
        """c_i <= 0 feasible. Raw (unscaled) margins in each metric's own units."""
        s = self.specs
        return np.array([
            m["overshoot"] - s.max_overshoot,
            m["e_hold_rms_deg"] - s.max_e_hold_rms_deg,
            m["h_peak_frac"] - s.max_h_peak_frac,
            m["sat_frac"] - s.max_sat_frac,
        ])

    def feasible(self, m):
        return (not m["diverged"]) and bool(np.all(self.constraints(m) <= 0.0))

    def outcomes(self, m):
        """Transformed outcomes for surrogate models: [log t_settle, c_1..c_4],
        with heavy-tailed constraints in log-ratio form (sign preserved: <=0 feasible)."""
        s = self.specs
        if m["diverged"]:
            m = dict(m)
        return np.array([
            np.log(max(m["t_settle"], 0.1)),
            np.log(m["overshoot"] + 0.5) - np.log(s.max_overshoot + 0.5),
            np.log(m["e_hold_rms_deg"]) - np.log(s.max_e_hold_rms_deg),
            m["h_peak_frac"] - s.max_h_peak_frac,
            m["sat_frac"] - s.max_sat_frac,
        ])

    def penalized(self, m):
        """Scalar penalty objective for CMA-ES / GA / random search ranking
        (feasible designs always rank above infeasible ones)."""
        o = self.outcomes(m)
        viol = np.clip(o[1:], 0.0, None).sum()
        if viol > 0 or m["diverged"]:
            return float(np.log(SLEW.horizon) + 1.0 + 10.0 * viol)
        return float(o[0])

    def evaluate(self, x):
        t0 = time.perf_counter()
        g = self.to_gains(x)
        m, _ = self.simulate(g)
        return {
            "x": np.asarray(x, dtype=float).tolist(),
            "gains": g.tolist(),
            "metrics": m,
            "constraints": self.constraints(m).tolist(),
            "outcomes": self.outcomes(m).tolist(),
            "objective": m["t_settle"],
            "feasible": self.feasible(m),
            "wall_time": time.perf_counter() - t0,
        }
