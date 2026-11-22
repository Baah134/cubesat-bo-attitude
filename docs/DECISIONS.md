# Decisions log

Dated log of design decisions, deviations from the project spec, and session notes.

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
   uses **Optuna TPE with `constraints_func`**, not a BoTorch GP. The project spec assumed BoTorch.
   Resolution: the IECON method is included as its own optimizer (`optim/tpe_iecon.py`,
   same formulation: objective = settling time, constraints c <= 0 via `constraints_func`).
   GP-based constrained BO in BoTorch (`optim/bo.py`) is added as the new method. Both
   are compared at equal budgets. Differences from the IECON setup: normalized log-gain
   unit cube for all methods; TPE startup = 2d+1 (same as BO's Sobol design) instead of 25.
2. **Equal budgets.** In the IECON code, GA (DE, maxiter=100, popsize=15) and CMA-ES
   (maxiter=100, popsize=15) were run for ~1,500 evaluations with penalty constraints, so
   the "4–20x sample efficiency" there compares unequal budgets. Here every method gets
   exactly the same evaluation budget (60 Stage A / 100 Stage B), as the project spec requires.
   The sample-efficiency gap may shrink; that will be reported as measured.
3. **MATLAB pidtune baseline added.** MATLAB R2024b with Control System Toolbox is
   installed here, so the "pidtune-style" analytic stand-in in the project spec (section 4.2) is
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

## Session 2 — 2026-11-11 (Stage A/B runs, expert baseline, large slew)

13. **BO acquisition switched from MC qLogNEI/qLogPOF to analytic LogCEI/LogPOF.** The MC
    version took ~57 s per acquisition optimization (100 evals x 10 seeds would not fit
    the day). Analytic LogConstrainedExpectedImprovement on a ModelListGP is ~1 s and is
    the standard single-point constrained EI. Same surrogate, same budget.
14. **Method `random` renamed `random_search`** (module name clashed with Python's
    `random`; the first Stage A random run crashed and was rerun, nothing else changed).
15. **Physics check on the best Stage A result (6.53 s, below the 6.63 s bang-bang
    rest-to-rest time).** Not a bug: t_settle is entry into the 0.1° band, and the best
    design enters the band at 0.34°/s and finishes decelerating inside it. Torque never
    exceeds 2 mN m in that run. Rest-to-rest and band-entry times are different quantities.
16. **Robust BO (bo_robust):** objective and constraints are the worst case over 8 training
    plants (TRAIN_SEED 555), so every evaluation costs 8 simulations but still counts as
    ONE evaluation against the 100 budget. This is stated wherever it is compared.
    Its "best t_settle" in the Stage B table is a worst-of-8 number and is NOT comparable
    to the nominal numbers; the 200-plant Monte Carlo (TEST_SEED 777, disjoint from 555) is
    the fair comparison.
17. **Expert baseline added (stage W):** Wie-Lu (1995) rate-limited eigenaxis quaternion
    feedback, u = -J(K sat_L(e) + C w) - Ki I + w x (J w + h), with direction-preserving
    torque saturation. Reason: a reviewer from ADCS would ask how tuned PID compares with
    the standard large-slew law, not only with pidtune and generic optimizers. Two
    variants: rule-designed with zero budget (`wie_rule`: w_max from the momentum spec with
    20% margin, wn from the same torque-limit rule as the analytic PID, zeta 0.9, ki 0.1)
    and BO-tuned over (wn, zeta, w_max, ki) with the same budget, specs and seeds as Stage
    B. This separates what the controller structure buys from what the tuning buys.
18. **Expert-law integrator gated to fine pointing (|e| < 2°).** First version without the
    gate never settled (integrator wound up over the 30° slew and unwound over > 100 s),
    the same failure as the analytic PID rule. Slew-mode / fine-pointing-mode switching is
    standard ADCS practice, so the gate is part of the expert design. The PID designs do
    NOT get this gate (they are the IECON controller structure); this favours the expert
    law and is stated in the results.
19. **Large-slew generalization test (90°, 200 s):** every tuned design is flown, without
    retuning, on a 90° slew on the nominal plant and on the 200 test plants. Chosen
    because the time-optimal (bang-bang) 90° slew about [1,1,1]/sqrt(3) already needs
    ~8.7 mN m s per wheel (87% of the 10 mN m s limit; 30° needs ~5 mN m s), so any
    controller that drives the body faster than the time-optimal profile hits the limit.
    (Corrected: a first draft said 11.5 mN m s, which is the single-axis number.)

