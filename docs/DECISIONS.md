# Decisions log

Dated log of design decisions, deviations from CLAUDE.md, and session notes.

## 2026-11-11 — Session 1

### Milestones
- **M0 Setup** — done. Repo, venv (Python 3.14.3; 3.11/3.12 not usable on this machine),
  dependencies locked in `requirements-lock.txt` (torch 2.15.0+cpu, botorch 0.18.1,
  gpytorch 1.15.2, optuna 5.0.0, cma 4.5.0, pymoo 0.6.2, numba 0.68.1, numpy 2.5.4, scipy 1.18.1).
- **M1 Stage A simulator + verification** — done. 6 Stage A tests + 5 metric tests pass.
  Sanity figure: `figures/sanity_stageA.png`.
- **M3 Stage B simulator + verification** — simulator and 11 tests written and passing
  (built early while Stage A optimizers were being smoke-tested).

### Decisions
1. **IECON code reuse.** Prince's IECON repo (github.com/Baah134/iecon-2026-pid-bo-tuning)
   uses **Optuna TPE with `constraints_func`**, not a BoTorch GP. CLAUDE.md assumed BoTorch.
   Resolution: the IECON method is included as its own optimizer (`optim/tpe_iecon.py`,
   same formulation: objective = settling time, constraints c <= 0 via `constraints_func`).
   GP-based constrained BO in BoTorch (`optim/bo.py`) is added as the new method. Both
   are compared at equal budgets. Differences from the IECON setup: normalized log-gain
   unit cube for all methods; TPE startup = 2d+1 (same as BO's Sobol design) instead of 25.
2. **Equal budgets.** In the IECON code, GA (DE, maxiter=100, popsize=15) and CMA-ES
   (maxiter=100, popsize=15) were run for ~1,500 evaluations with penalty constraints, so
   the "4–20x sample efficiency" there compares unequal budgets. Here every method gets
   exactly the same evaluation budget (60 Stage A / 100 Stage B), as CLAUDE.md requires.
   The sample-efficiency gap may shrink; that will be reported as measured.
3. **MATLAB pidtune baseline added.** MATLAB R2024b with Control System Toolbox is
   installed here, so the "pidtune-style" analytic stand-in in CLAUDE.md §4.2 is
   complemented by the real `pidtune` (scripts/pidtune_baseline.m → configs/pidtune_gains.json).
   Plant per axis: 1/(J s^2 (T_m s+1)) with a Ts/2 = 0.05 s input delay for the 10 Hz ZOH.
   pidtune result: crossover 5.9 rad/s, PM 60°, for both axis groups. The analytic rule is kept too.
4. **Hold-RMS spec calibrated from 0.02° to 0.002°.** Sobol sweep of 512 Stage A designs
   (`results/calibration_A.json`): every design that completes the slew already meets
   0.02° (settled-design hold RMS 5th–90th percentile: 0.0004°–0.0064°), so 0.02° never
   binds. At 0.002° the feasible fraction drops from 10.7% to 5.7% and the spec trades
   against slew speed (higher gains pass more sensor noise to the pointing).
5. **Integration step 0.01 s (100 Hz), controller 10 Hz ZOH.** Halving dt changes t_settle
   by ≤ 0.02 s, overshoot by < 0.01 pp, peak momentum by < 0.1% (tests).
6. **Anti-windup:** integrator clamp |Ki·I| ≤ 0.25·tau_max plus conditional integration
   (no integration while saturated in the direction that deepens saturation).
   Derivative term acts on measured gyro rate (rate reference zero), not on error.
7. **Saturation metric:** fraction of slew controller samples with any axis saturated.
8. **Energy metric:** mechanical work at the wheels, ∫ Σ|tau_i · omega_w,i| dt with
   omega_w = h/J_w (wheel speed relative to the body). Reported, not constrained.
9. **Momentum limit:** when |h_i| ≥ h_max and the commanded torque would increase |h_i|,
   that wheel's torque is zeroed (state clamped). Any such event = momentum violation.
10. **Disturbances** applied in body axes (bias + orbital-period sinusoid with random phase
    + ZOH random component). A simplification of gravity-gradient/drag/SRP torques.
11. **Diverged runs:** metrics clipped to caps (t_settle = horizon, overshoot 200%, hold
    RMS 10°, fractions 1.0) and marked infeasible; never NaN/inf to any optimizer.
12. **Constrained BO before first feasible point:** maximize qLogProbabilityOfFeasibility
    (BoTorch warns that LogNEI is a poor choice when all points are infeasible).

### Findings so far
- **MATLAB pidtune is feasible on Stage A** (t_settle 36.3 s, overshoot 0.9%, hold RMS
  0.00042°, saturated 2% of the slew). Because the derivative acts on gyro rate, after the
  initial saturation the high Kd rate-limits the slew and the anti-windup contains the
  integrator. It is therefore a strong baseline here, unlike in the IECON plants.
- **Analytic rule is infeasible** (overshoot 14%, does not settle in 120 s): integrator
  windup during the slew leaves a slow tail.

### Next
- Smoke-test all optimizers on Stage A, then run Stage A (7 methods × 10 seeds × 60 evals).
