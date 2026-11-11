"""Default physical and simulation parameters (see docs/EXPLAINER.md for reasoning).

All SI units: kg, m, s, rad, N*m, N*m*s.
"""
from dataclasses import dataclass, field, replace
import numpy as np

ARCSEC = np.pi / (180.0 * 3600.0)
DEG = np.pi / 180.0


@dataclass(frozen=True)
class SpacecraftParams:
    # 3U CubeSat, ~4 kg, 10x10x34 cm uniform box. Principal inertia (kg m^2).
    J: tuple = (0.042, 0.042, 0.0067)
    # Small products of inertia (off-diagonal), zero nominally; used in Monte Carlo.
    J_offdiag: tuple = (0.0, 0.0, 0.0)  # (Jxy, Jxz, Jyz)
    tau_max: float = 2.0e-3        # wheel torque limit, N m
    h_max: float = 10.0e-3         # wheel momentum limit, N m s
    J_wheel: float = 1.6e-5        # wheel rotor inertia, kg m^2 (h_max <-> ~6000 rpm)
    T_m: float = 0.03              # wheel motor torque time constant, s

    @property
    def J_matrix(self):
        Jxx, Jyy, Jzz = self.J
        Jxy, Jxz, Jyz = self.J_offdiag
        return np.array([[Jxx, Jxy, Jxz], [Jxy, Jyy, Jyz], [Jxz, Jyz, Jzz]], dtype=float)


@dataclass(frozen=True)
class EnvParams:
    # Disturbance torque per axis: bias + orbital-rate sinusoid + random component.
    dist_bias: tuple = (5.0e-7, -3.0e-7, 2.0e-7)   # N m
    dist_sin_amp: tuple = (3.0e-7, 3.0e-7, 1.5e-7)  # N m
    orbit_period: float = 95.0 * 60.0               # s
    dist_rand_std: float = 1.0e-7                   # N m, per controller step (ZOH)
    dist_scale: float = 1.0                         # Monte Carlo multiplier
    att_noise_std: float = 10.0 * ARCSEC            # star-tracker-like, rad (1 sigma)
    gyro_noise_std: float = 1.0e-5                  # rad/s (1 sigma, per sample)
    gyro_bias: tuple = (2.0e-6, -1.5e-6, 1.0e-6)    # rad/s


@dataclass(frozen=True)
class SimParams:
    dt: float = 0.01          # RK4 integration step (100 Hz); see step-convergence test
    ctrl_dt: float = 0.1      # controller period (10 Hz, zero-order hold)


@dataclass(frozen=True)
class Scenario:
    name: str
    angle_deg: float          # slew angle (or initial offset for hold)
    axis: tuple               # rotation axis (unit vector, body frame); Stage A uses axis index 0
    horizon: float            # s
    kind: str                 # "slew" or "hold"
    transient: float = 0.0    # s excluded from hold RMS


SLEW = Scenario("slew", 30.0, tuple(np.ones(3) / np.sqrt(3)), 120.0, "slew")
HOLD = Scenario("hold", 0.5, tuple(np.ones(3) / np.sqrt(3)), 600.0, "hold", transient=150.0)
LARGE_SLEW = Scenario("large_slew", 90.0, tuple(np.ones(3) / np.sqrt(3)), 200.0, "slew")

__all__ = ["SpacecraftParams", "EnvParams", "SimParams", "Scenario", "SLEW", "HOLD",
           "LARGE_SLEW", "ARCSEC", "DEG", "replace", "field"]
