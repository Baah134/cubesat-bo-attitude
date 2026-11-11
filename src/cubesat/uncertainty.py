"""Plant uncertainty sampling for robustness studies (CLAUDE.md section 5).

Each sample perturbs:
  - principal inertia: each axis x U(0.7, 1.3)                (+-30%)
  - products of inertia: each U(-0.02, 0.02) * min principal  (small, keeps J positive definite)
  - disturbance magnitude: all components x U(0.5, 3.0)
  - motor time constant: x U(0.5, 1.5)                        (+-50%)
  - new sensor-noise / disturbance-phase seeds
The same sample set (same seed) is used for every method (common random numbers).
Test plants (seed 777) are disjoint from the plants used by robust BO (seed 555).
"""
from dataclasses import dataclass

import numpy as np

from .params import EnvParams, SpacecraftParams, replace

TEST_SEED = 777
TRAIN_SEED = 555


@dataclass(frozen=True)
class PlantSample:
    index: int
    sc: SpacecraftParams
    env: EnvParams
    slew_seed: int
    hold_seed: int
    factors: dict


def sample_plants(n, seed=TEST_SEED, sc0=SpacecraftParams(), env0=EnvParams()):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        f_J = rng.uniform(0.7, 1.3, 3)
        J = tuple(float(j * f) for j, f in zip(sc0.J, f_J))
        off = tuple(float(x) for x in rng.uniform(-0.02, 0.02, 3) * min(J))
        f_d = float(rng.uniform(0.5, 3.0))
        f_T = float(rng.uniform(0.5, 1.5))
        sc = replace(sc0, J=J, J_offdiag=off, T_m=sc0.T_m * f_T)
        assert np.all(np.linalg.eigvalsh(sc.J_matrix) > 0)
        env = replace(env0, dist_scale=env0.dist_scale * f_d)
        seeds = rng.integers(0, 2 ** 31 - 1, 2)
        out.append(PlantSample(i, sc, env, int(seeds[0]), int(seeds[1]),
                               {"J_factor": f_J.tolist(), "J_offdiag": list(off),
                                "dist_factor": f_d, "T_m_factor": f_T}))
    return out
