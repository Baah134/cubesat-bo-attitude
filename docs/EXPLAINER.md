# EXPLAINER — study guide for Prince

Written for someone who knows control and BO but not spacecraft dynamics.
Results-dependent sections (marked **[after results]**) are filled in from
`RESULTS.md` once the experiments are frozen; every number there comes from
`results/*.json`.

---

## 1. How a reaction-wheel CubeSat points itself

A CubeSat has no thrusters for fine pointing. It turns by spinning **reaction
wheels**: small flywheels driven by motors. When the motor applies torque `τ`
to a wheel, the wheel applies `−τ` to the spacecraft body (Newton's third law
for rotation). Spin a wheel up one way and the body turns the other way.

The key invariant: with no external torque, **total angular momentum (body +
wheels) is constant**. The wheels don't create momentum, they *store* it. That
is why the test `test_internal_wheel_torque_conserves_inertial_momentum` is the
single most important correctness check in this repo: if it failed, the
simulator would be inventing momentum.

Two hard limits follow:
- **Torque limit** `τ_max = 2 mN·m`: how fast a wheel can be accelerated. It
  caps how fast the body can start/stop turning.
- **Momentum limit** `h_max = 10 mN·m·s` (≈ 6000 rpm for this wheel inertia):
  the wheel can't spin faster. A slew needs momentum to get the body rotating,
  and small external disturbances (gravity gradient, drag, solar pressure)
  push momentum into the wheels continuously over an orbit.

**Momentum saturation** is when a wheel is at its speed limit: it can no longer
produce torque in that direction, so the controller loses authority on that
axis. That's why there's a spec keeping peak momentum ≤ 90% of `h_max`
(a margin for disturbances). Real missions periodically **dump momentum**
using magnetorquers (coils that push against Earth's magnetic field). That is
*not modeled* here; our horizons (≤ 600 s) are short compared with the time
disturbances need to fill the wheels (~hours at 1e−6 N·m).

## 2. Equations (what the simulator integrates)

Stage A (one axis, `src/cubesat/single_axis.py`):

    J θ̈ = τ_eff + τ_d          body
    ḣ    = −τ_eff               wheel momentum
    τ̇    = (u_sat − τ)/T_m      motor lag, T_m = 30 ms

Stage B (three axes, `src/cubesat/three_axis.py`):

    q̇ = ½ q ⊗ [0, ω]                              attitude kinematics
    J ω̇ = −ω × (J ω + h) + τ_eff + τ_d              Euler's equation with wheels
    ḣ = −τ_eff,   τ̇ = (u_sat − τ)/T_m

The `−ω × (Jω + h)` term is **gyroscopic coupling**: when the body rotates
about a non-principal axis (our slew axis is [1,1,1]/√3), the axes interact.
This is what Stage B adds over three independent copies of Stage A.

Integration: fixed-step RK4 at 100 Hz; controller at 10 Hz with zero-order
hold (the torque command is held constant between samples, as in real flight
software). Halving the step changes the metrics by < 0.02 s (tested).

## 3. Quaternions, error quaternions and unwinding

A **unit quaternion** `q = [q0, q1, q2, q3]` (scalar first in this repo)
encodes a rotation by angle `φ` about unit axis `a` as
`q = [cos(φ/2), sin(φ/2)·a]`. Four numbers, no singularities (unlike Euler
angles), one constraint `|q| = 1`, which we restore by renormalizing every
step (`test_quaternion_norm_long_run` checks it stays within 1e−9).

The **error quaternion** `q_e = q_ref⁻¹ ⊗ q` is the rotation from where we want
to be to where we are. Its vector part, times 2, is approximately the error
rotation vector, which is what the PID acts on.

**Unwinding:** `q` and `−q` describe the *same* attitude. If the controller
uses `q_e` naively, a 350° error (really −10°) can make it rotate the long way
round, 350°. Using `sign(q_e0)·q_e_vec` always picks the short way.
`test_error_quaternion_no_unwinding` checks exactly this case.

## 4. Why these parameter values

| Parameter | Value | Reasoning |
|---|---|---|
| Inertia | diag(0.042, 0.042, 0.0067) kg·m² | 4 kg uniform 10×10×34 cm box: `J = m(b²+c²)/12` |
| τ_max | 2 mN·m | typical CubeSat-class wheels are ~1–6 mN·m |
| h_max | 10 mN·m·s | typical small wheels are ~5–30 mN·m·s |
| J_wheel | 1.6e−5 kg·m² | gives ~6000 rpm at h_max |
| T_m | 30 ms | motor current loop + driver, small compared with 100 ms control period |
| Disturbance | ~1e−7 to 1e−6 N·m | typical LEO environmental torques for a 3U |
| Star tracker | 10 arcsec (1σ) | typical CubeSat star tracker cross-boresight |
| Controller rate | 10 Hz | common ADCS loop rate |

These are representative values, not a specific spacecraft. A reviewer may ask
"which mission?" The honest answer: a generic 3U, with values chosen inside
published ranges for CubeSat components. (Check vendor datasheets, e.g. small
reaction wheels from several CubeSat suppliers, before quoting specific ones.)

## 5. Why each spec exists (mission view)

- **Settling time (objective):** time to finish a slew and be usable. Shorter
  means more imaging targets per pass or faster ground-station acquisition.
- **Overshoot ≤ 5%:** overshooting a 30° slew by > 1.5° can sweep sensors past
  keep-out zones (e.g. pointing an instrument toward the Sun).
- **Hold RMS ≤ 0.002° (7.2 arcsec):** pointing stability for imaging/comms.
  Calibrated so it actually binds (see DECISIONS.md #4): fast controllers pass
  more star-tracker noise into the pointing, so speed trades against stability.
- **Peak momentum ≤ 90%:** keeps margin so disturbances can't saturate a wheel
  mid-maneuver.
- **Saturated ≤ 20% of the slew:** some saturation during large slews is normal
  (you want to use the available torque), but a controller that is saturated
  most of the time is effectively bang-bang and hard to predict.

## 6. The controller and anti-windup

Per axis: `u = −(Kp·e + Kd·ω_meas + Ki·∫e)`, clipped to ±τ_max. Derivative acts
on the **measured gyro rate**, not the derivative of the error, so there's no
"derivative kick" at the start of a slew.

**Integrator windup** is the main nonlinear effect you'll see: during a 30°
slew the error is large for many seconds, the integrator accumulates, and
after the body arrives the integrator has to "unwind", producing overshoot and
a slow tail. We limit it two ways: clamp `|Ki·I| ≤ 0.25 τ_max`, and don't
integrate while saturated in the direction that deepens saturation. Even so,
designs with a slow integrator show long tails, see `figures/sanity_stageA.png`
(analytic rule). Many real ADCS designs disable the integrator during large
slews; that is a possible extension, not done here.

## 7. How constrained BO works here

1. **Search space:** gains in log10 space, normalized to [0,1]^d. Gains span
   orders of magnitude; in log space a GP length-scale means "a factor of X in
   gain", which is physically sensible. Bounds (`BOUNDS_A`, `BOUNDS_B`) span
   from clearly sluggish (natural frequency ≪ 0.01 rad/s) to clearly unstable
   at 10 Hz sampling (derivative gain so large the loop rings).
2. **One GP per outcome:** objective `log t_settle` plus four constraint
   margins `c_i` (≤ 0 = feasible). Log transforms tame heavy tails: overshoot
   and hold RMS range over several orders of magnitude.
3. **Acquisition:** constrained expected improvement:
   `EI(x) × Π_i P(c_i(x) ≤ 0)`. EI says "how much better than the best
   feasible design so far, in expectation"; the product of feasibility
   probabilities discounts points the GPs think will violate a spec. Computed
   in log space (LogCEI) for numerical stability.
4. **Before any feasible point exists** there's no incumbent for EI, so we
   maximize the probability of feasibility instead.
5. **Initial design:** 2d+1 scrambled Sobol points (7 for Stage A, 13 for B).

**Difference from the IECON paper:** IECON used Optuna's **TPE** (a density-
ratio model, not a GP) with a constraints function. Both are Bayesian
optimization in the broad sense; TPE is included here as "the IECON method"
so you can say how the new GP-based method compares with your published one.

## 8. Baselines and what "fair" required

- **MATLAB pidtune:** the textbook industry tool, as in IECON. Tuned on the
  linear plant `1/(J s²(T_m s+1))` with a Ts/2 delay for the ZOH; it targets
  60° phase margin and does not know about saturation or momentum limits.
- **Analytic rule:** `Kp = τ_max/θ_slew`, ζ = 0.9, Ki a decade below ωn.
- **Random search (Sobol), CMA-ES, GA:** standard black-box baselines.
- Fairness: identical simulator, specs, nominal noise seeds and **identical
  evaluation budget** for every optimizer; 10 optimizer seeds each; best
  feasible = lowest t_settle among designs meeting *all* specs, never the best
  infeasible point.

## 9. What the results show / do not show **[after results]**

## 10. Likely reviewer questions **[after results]**
