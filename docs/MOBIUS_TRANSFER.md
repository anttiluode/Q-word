# Q-word v3 — Möbius transfer: reversible operations are not sufficient memories

> **2026-10-09:** the learned-model ranking below is superseded by [the v3 check](MOBIUS_TRANSFER_CHECK.md). The physics filter is the Bayes ceiling; the qubit model's lead disappears with 3,200 training sequences; and the original GRU's readout could not depend on the action. The protocol, receipt and numbers below are unchanged.

**2026-10-08. Status: exploratory synthetic study, not preregistered.** This is an independent implementation motivated by [MovingTarget2's Möbius addendum](https://github.com/anttiluode/MovingTarget2/blob/main/MOBIUS.md), not an import of its frozen benchmark or a replication of its twenty seeds. MovingTarget2 is unchanged. No quantum hardware is involved.

## Scientific question

Q-word v2's quantum-constrained recurrence won its planted quantum-instrument generator. Does its geometric constraint help on **independently specified classical oscillator dynamics**? Separate three distinct concepts: (1) a known reversible action history; (2) invariants of a physical constellation; and (3) a predictive state given noisy, incomplete observations.

## Three environments, identical observation interface

Each trajectory draws a fresh, *unseen* constellation of 16 angles from a disclosed von Mises mixture: mean direction uniform in `[-pi, pi]`, concentration uniform in `[0.12, 2.2]`. Every step has one of **three observed actions**, phases `0`, `2pi/3`, `4pi/3`. The simulator has three worlds:

- **M: Möbius.** Exact time-0.11 flow `dtheta/dt=sin(phi-theta)`. A known sequence of actions is exactly a composable `SU(1,1)` map. Cross-ratios are fixed.
- **H: second harmonic.** Time-0.11 flow `dtheta/dt=sin(phi-theta)+0.28 sin(2(phi-theta))`, RK4 with 12 substeps. Cross-ratios can change. No exact `SU(1,1)` odometer exists in general.
- **D: hidden drift.** Exact Möbius read, then independent Gaussian phase noise of standard deviation 0.075 radians per oscillator. Even a perfect odometer of *known reads* is blind to the stochastic drift.

At each time **before** the action changes the phases, the world samples the same binary observation interface:

\[
p(y_t=1\mid\theta_t,a_t)=\operatorname{clip}_{[0.04,0.96]}\bigg(\frac12+0.37\operatorname{Re}(e^{-i\phi_a}m_1)+0.09\operatorname{Re}(e^{-2i\phi_a}m_2)\bigg), \quad m_j=\tfrac1{16}\sum_u e^{ij\theta_u}.
\]

**Learners see only current action, earlier actions, and earlier binary observations.** The current observation is revealed *after* the prediction. No initial phases or latent cross-ratios are supplied. Train/validation/test all draw independent initial constellations and outcome noise. The held-out constellations differ across splits. The readout is a newly designed finite-information test, not MovingTarget2's exact phase-change software listener. Direct numeric scores across the two repositories cannot be compared.

## Baselines and access accounting

| Method | What it is allowed to know | What is evaluated |
| --- | --- | --- |
| Hidden-state oracle | Initial phases, every hidden phase update, source law | Privileged lower reference for test NLL; **not** a fair observation-only learner |
| Physics particle filter | Actions, past y, **true world law**, disclosed initial-shape *distribution*, 48 latent particles; no actual phases | Matched-observation-access, **model-informed** baseline (more prior knowledge than learners) |
| One-step tabulation | Training sequences, current/previous action and previous bit | Short-history heuristic |
| Q-word qubit instrument | Training sequences; three real Bloch-state coordinates; qubit constraints | Learned predictive NLL |
| Real-operator recurrence | Same training sequences; three real state coordinates; learned nonlinear operator | Learned predictive NLL |
| GRU | Same sequences; four persistent coordinates | Learned predictive NLL |
| Windowed transformer | Same sequences; causal eight-token attention window | Learned predictive NLL |

Particle filtering resamples when the effective sample size drops below half the population. It propagates posterior particles by the correct world transition *after* using `y_t` for the posterior update. It never gets the hidden simulator trajectory. It is **not parameter-matched** to any recurrent model and its exact prior/transition law is a major advantage.

Equal steps do not give equal compute, memory or information rights: the neural models retain Q-word v2's existing nonmatched parameter counts (33 / 87 / 144 / 1713). This is **not** a quantum computational advantage test, and the transformer is intentionally small and unpretrained.

## Canonical exploratory receipt

Run seeds 2026, 2027, 2028 separately for independent initial shapes and outcomes, per world, with **160 train, 35 validation, 80 test sequences, length 24, 65 Adam updates per model, minibatch 40, 48-particle physics filter**. Learning rates are unchanged from Q-word v2 (qubit 0.012, others 0.006). The results were inspected during development; no threshold was frozen before testing. The saved JSON is [`results/mobius-v3.json`](../results/mobius-v3.json).

**Mean test NLL, nats/token, lower better:**

| Model | Möbius | Harmonic | Hidden drift |
| --- | ---: | ---: | ---: |
| Privileged full-state oracle | 0.64349 | 0.64231 | 0.65395 |
| Physics filter (correct law, 48 particles) | **0.67511** | **0.67396** | **0.68093** |
| Qubit-inspired recurrence | 0.68242 | 0.68158 | 0.68920 |
| Real-operator recurrence | 0.68791 | 0.68482 | 0.68715 |
| GRU | 0.69258 | 0.69270 | 0.69400 |
| Eight-token transformer | 0.69439 | 0.69439 | 0.69673 |
| One-step tabulation | 0.69258 | 0.69246 | 0.69287 |
| Uniform prediction | 0.69315 | 0.69315 | 0.69315 |

**Interpretation:** the true-law observation-only physics filter wins on all three worlds. The qubit-constrained recurrence beats the other learned models on M/H at this limited budget, but loses to the real-operator recurrence on hidden drift. This is *consistent* with a geometric inductive bias benefiting a structured generator and failing under unmodelled changes, but **three seeds do not establish that explanation or a reliable ranking**. It could be training speed or hyperparameters. The gap from oracle reflects both learning limitations and information inaccessible to an observation-only agent.

## Exact geometry checks (not learned advantages)

Thirty-two pings applied to a distinct sixteen-phase constellation were composed into one two-by-two `SU(1,1)` matrix. Across all three diagnostic seeds, group reconstruction and inverse errors are at most `7.2e-15` radians. Möbius cross-ratio change is at most `5.6e-15` on the arctangent scale. A second harmonic changes the median arctan-cross-ratio by `0.0041–0.0168`; an inverse of known actions applied despite unseen drift leaves `0.41–0.51` radians RMS error. These are this repository's local numerical demonstrations, *not* the exact MovingTarget2 figures, which used six groups and different read protocols.

An independent four-phase analytic pair has the same `m1` (gap `1.2e-16`) but different `m2` (gap 1.5). Applying the same exact pulse at duration 0.25 separates their next `m1` by approximately 0.164. Thus a compact answer representation need not be a sufficient state for future updates.

## A distinction worth keeping

Represent a constellation as its initial shape `C` and a known transformation `G_t`. The idealized action process factors as

\[
C_{t+1}=C_t,\quad G_{t+1}=M(a_t)G_t,\quad p_t=F(C,G_t,a_t).
\]

For exact Möbius dynamics this is a useful **coordinate decomposition**, not evidence that `G_t` alone predicts future responses across unknown shapes. A tracker that knows `C` can recover full phases from `G_t` and that initial reference; the 3-number odometer alone cannot. Harmonic forcing changes `C`, and stochastic drift makes even `G_t` uncertain. The prior over `C`, observation noise and stochastic drift are why a predictive-state filter must maintain uncertainty.

`SU(1,1)` oscillator maps and `SU(2)` qubit rotations are **different noncommuting groups**. They share operator composition and dimensionality, not interchangeable physical constraints. Q-word's learned instrument is fully classical code. The physics filter's win supports state estimation with accurate transition knowledge, not quantum memory advantage.

## Follow-up falsifiers

1. Train every model to validation-tuned convergence under matched GPU/CPU cost, parameter count, and longer windows. If the qubit model's edge disappears, the current advantage was likely an optimization/short-budget artifact.
2. Compare true-law 48-particle filtering with learned-transition particle filtering or a state-space model with comparable **system identification** budget. Its privileged law currently makes its win unsurprising.
3. Provide calibrated probes of the initial constellation with matched observation budget, then test delayed strong probes. If every model is unable to identify shape without informative measurements, that is an observability limit, not failure of memory.
4. Sweep second-harmonic strength and hidden-drift variance *without retraining* to test whether performance tracks violations of group closure rather than mere noise/temperature.
5. Test held-out phase-port geometries and action sequences. This run has three fixed query ports, not arbitrary continuous queries.

## Reproduction

```bash
pip install -e '.[sequence]'
python -m unittest discover -s tests -v
python -m qword.mobius --worlds mobius harmonic drift --seeds 2026 2027 2028 \
  --train 160 --valid 35 --test 80 --length 24 --steps 65 --batch 40 --particles 48 \
  --output results/mobius-v3.json
```

The included [`web/mobius.html`](../web/mobius.html) is an offline viewer of the committed receipt, not an independent second run.
