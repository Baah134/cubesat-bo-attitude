# Results (simulation only)

Every number below is from the committed histories in `results/` and regenerates with
`make figures` (or `python reproduce.py figures`). Full tables: `results/tables/`.
10 optimizer seeds per method; equal evaluation budgets inside a stage; fixed designs
(pidtune, analytic rule, Wie-Lu rule) use zero budget.

Specs (all must hold): slew overshoot ≤ 5%; pointing-hold RMS ≤ 0.002°; peak wheel
momentum ≤ 90% of 10 mN·m·s; torque saturated in ≤ 20% of slew controller samples.
Objective: time to settle within 0.1° after a 30° slew about [1,1,1]/√3.

## 1. Stage A — single axis, PID, 60 evaluations

| Method | Runs finding a feasible design | Best feasible t_settle (mean ± std) |
|---|---|---|
| Constrained BO (GP) | 100% | 7.74 ± 0.62 s |
| TPE (IECON method) | 100% | 9.28 ± 1.15 s |
| CMA-ES | 100% | 11.46 ± 3.74 s |
| Random search (Sobol) | 100% | 14.11 ± 4.72 s |
| GA | 90% | 39.89 ± 41.04 s |
| MATLAB pidtune (fixed) | — | 36.33 s |
| Analytic rule (fixed) | — | infeasible (14% overshoot) |

## 2. Stage B — three axes, PID (6 gains), 100 evaluations

| Method | Runs finding a feasible design | Best feasible t_settle (mean ± std) | Evals to first feasible (median) |
|---|---|---|---|
| Constrained BO (GP) | 100% | 7.26 ± 0.61 s | 19 |
| TPE (IECON method) | 70% | 10.09 ± 6.12 s | 34 |
| CMA-ES | 90% | 32.27 ± 36.43 s | 38 |
| GA | 10% | 9.08 s (one run) | 46 |
| Random search (Sobol) | 60% | 54.36 ± 45.57 s | 28 |
| MATLAB pidtune (fixed) | — | 48.03 s | — |
| Analytic rule (fixed) | — | infeasible (11.5% overshoot) | — |
| Robust BO (worst of 8 training plants)* | 90% | 10.02 ± 3.64 s (worst-of-8) | 33 |

\*Robust BO's number is a worst case over 8 plants and is not comparable to the
nominal rows; Section 4 is the fair comparison.

## 3. Expert baseline — Wie-Lu rate-limited eigenaxis law, same plant and specs

| Design | Runs finding a feasible design | Best feasible t_settle | Evals to first feasible (median) |
|---|---|---|---|
| Rule-designed (zero budget) | — | 37.27 s | — |
| BO-tuned (4 parameters, 100 evals) | 100% | 5.54 ± 0.33 s | 1.5 |

The integrator of the expert law runs only within 2° of the target (fine-pointing mode);
the PID designs do not get this gate. This favours the expert law (DECISIONS.md #18).

## 4. Robustness — 30° slew + hold on 200 sampled plants (identical for every design)

Inertia ±30%, products of inertia ±2%, disturbance ×0.5–3, motor lag ±50%, new noise.
Compliance = fraction of plants on which a design meets all four specs.

| Design | Compliance (mean ± std over seeds) | Worst seed | t_settle median / 95th pct | Main failure |
|---|---|---|---|---|
| MATLAB pidtune [PID] | 100% | 100% | 48.0 / 48.1 s | none |
| Robust BO [PID] | 95.2 ± 7.6% | 77% | 8.2 / 19.6 s | overshoot, hold (2% each) |
| Constrained BO [PID] | 86.2 ± 16.4% | 45.5% | 8.1 / 10.5 s | hold RMS (13.5%) |
| TPE (IECON) [PID] | 84.2 ± 15.9% | 56% | 8.5 / 24.2 s | hold RMS (14%) |
| CMA-ES [PID] | 82.1 ± 22.1% | 37.5% | 14.3 / 120 s | hold RMS (16%) |
| Expert law, BO-tuned | 79.3 ± 17.3% | 49.5% | 7.6 / 8.9 s | overshoot (19%) |
| Random search [PID] | 69.1 ± 36.2% | 14% | 37.4 / 120 s | hold RMS (31%) |
| Expert law, rule-designed | 61.0% | 61% | 39.0 / 44.3 s | hold RMS (39%) |
| GA [PID] (1 design) | 31.5% | 31.5% | 10.7 / 12.0 s | hold RMS (69%) |

## 5. Generalization — 90° slew never seen during tuning, 200 plants

Even the time-optimal (bang-bang) 90° slew needs ~8.7 mN·m·s per wheel, 87% of the
10 mN·m·s capacity, so a controller that drives the body faster than that profile hits
the wheel limit. The BO-tuned PID designs peak near 34°/s on the nominal plant.

| Design | Nominal t_settle (median) | Overshoot ≤ 5% (plants) | Wheel momentum hit limit (plants) |
|---|---|---|---|
| Expert law, BO-tuned | 13.5 s | 100% | 0% |
| Expert law, rule-designed | 37.4 s | 100% | 0% |
| Constrained BO [PID] | 18.0 s | 3.6% | 81.2% |
| Robust BO [PID] | 18.5 s | 11.1% | 68.3% |
| TPE (IECON) [PID] | 18.7 s | 0% | 80.6% |
| CMA-ES [PID] | 20.0 s | 66.5% | 54.8% |
| MATLAB pidtune [PID] | 51.9 s | 100% | 0% |

## 6. What this supports

1. **Among optimizers, GP-based constrained BO is the most sample-efficient and reliable
   here.** It is the only optimizer that found a feasible PID design on every seed in
   Stage B, and it has the lowest mean and variance. At equal budgets the margin over
   TPE (the IECON method) is modest in Stage A (7.7 vs 9.3 s) and larger in Stage B.
2. **Tuning for nominal speed costs robustness, and robust BO buys most of it back.**
   Worst-of-8 training raised mean compliance from 86% to 95% with almost no change in
   median t_settle. One robust-BO seed found no feasible design, and its 95th-percentile
   t_settle is worse than nominal BO's.
3. **Controller structure matters more than the optimizer once the slew changes.** PID
   gains tuned on 30° saturate the wheels on most plants at 90°. The rate-limited
   expert law, tuned by the same BO on the same 30° problem, does not, and is faster.
4. **Expert structure plus BO is the fastest nominal design (5.5 s) but not the most
   robust on 30°** (79% compliance, overshoot under inertia error). Neither structure
   dominates on every axis.
5. **MATLAB pidtune is slow but never fails** in these tests: the conservative baseline
   an operator might still prefer.

## 7. What this does NOT show

- Anything about hardware: no HIL, no flight software, no real sensors.
- Real ADCS: no attitude estimator (the controller sees noisy measurements directly),
  generic body-frame disturbances instead of modelled gravity-gradient/drag/SRP, no
  momentum dumping, three orthogonal wheels, no failure modes.
- That BO wins in general: one spacecraft, one objective, budgets of 60–100 evaluations.
- Onboard tuning: all tuning is offline, at design time.
- Robust BO was not run on the expert law (budget/time), so "robust BO + expert law"
  is untested.
