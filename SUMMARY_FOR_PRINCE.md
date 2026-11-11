# Summary for Prince — Project 1 (facts only; write your own words from these)

**What it is.** A simulation study of reaction-wheel attitude control for a 3U-class
CubeSat. Six optimizers tune the controller under four spacecraft specs at equal
budgets (10 seeds each). Baselines: MATLAB pidtune, a textbook rule, and the Wie-Lu
(1995) rate-limited eigenaxis law used for large slews. Designs are tested on 200
perturbed spacecraft and on a 90° slew not used in tuning.

**What you built (with Claude Code).** Python simulator (single-axis and three-axis
quaternion dynamics, wheel momentum limits, motor lag, noise, disturbances); 31 tests
including conservation-law checks; constrained BO in BoTorch; robust BO; four other
optimizers; MATLAB pidtune baseline; Monte Carlo robustness and generalization tests;
figures and tables reproducible from one command.
Repo (private): https://github.com/Baah134/cubesat-bo-attitude

**Key numbers (simulation).**
- Three-axis PID, 100 evals: constrained BO 7.26 ± 0.61 s settling, feasible on 10/10
  seeds; TPE (IECON method) 10.09 ± 6.12 s, 7/10; MATLAB pidtune 48.0 s.
- 200-plant robustness: robust BO 95% spec compliance vs 86% for nominal BO, at similar
  median settling (8.2 vs 8.1 s). pidtune 100% but 6x slower.
- 90° slew (unseen): BO-tuned PID hit the wheel momentum limit on 81% of plants and
  exceeded 5% overshoot on 96%; the BO-tuned Wie-Lu law hit it on 0%, met overshoot on
  100%, and settled faster (13.5 vs 18.0 s nominal).
- BO-tuned Wie-Lu law: fastest 30° design (5.54 ± 0.33 s) but 79% compliant on the
  200-plant test.

**One-sentence takeaway you can verify.** Constrained BO was the most reliable tuner, but
on a larger slew the controller's structure (rate limiting) mattered more than which
optimizer tuned it.

**Words to avoid.** "flight-tested", "onboard", "real-time self-tuning", "validated on
hardware", "BO is always better". Say "in simulation", "offline tuning".

**Your skills shown.** Spacecraft attitude dynamics and quaternions; control design
under actuator constraints; Bayesian optimization; Monte Carlo robustness analysis;
MATLAB Control System Toolbox; experimental fairness (equal budgets, seeds, held-out
test plants); reproducible research.

**Things to be able to explain in an interview.** Why quaternion error uses sign(qe0)
(unwinding); why the integrator needs anti-windup; why BO uses one GP per constraint;
why robust BO trains on 8 plants and tests on 200 different ones; why 90° hits the
momentum limit (time-optimal profile already uses 87% of wheel capacity; un-rate-limited
PID overshoots that rate). All in docs/EXPLAINER.md.
