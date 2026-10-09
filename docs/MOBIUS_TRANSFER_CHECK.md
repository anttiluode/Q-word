# Q-word v3 check — the ceiling, the training-set size and the GRU readout

**2026-10-09. Post-hoc check of an exploratory study, not preregistered.** These runs were chosen after reading v3's receipt and scratch reruns, so they test v3's own listed falsifiers rather than independent predictions. Every number below comes from committed code and receipts. Nothing in v1–v3 changes: the original four models, `models()` and both original runners are untouched, so [`results/mobius-v3.json`](../results/mobius-v3.json) and the v2 receipts still reproduce.

## Short version

- **The physics filter is the ceiling.** v3's 48-particle filter scores within 0.001–0.002 nats/token of a 5,000-particle Bayes filter, and 5,000 particles agree with 1,000 and 20,000 in the Möbius world. No learner that sees only actions and outcomes can do better on average. About 60% of the hidden-state oracle's lead is information the observations never carry.
- **v3 does test memory.** Knowing the dynamics and prior but never using the outcomes (open loop) reaches only 19–30% of the way from uniform to the ceiling.
- **The qubit model's lead was a small-training-set effect.** At the published 160 training sequences it stays the best learner even after 3,000 updates, because the larger models overfit within a few hundred updates. With 3,200 training sequences the real-operator recurrence beats it in 8 of 9 world–seed pairs.
- **The original GRU could not compete.** Its prediction is `sigmoid(w·r + v_a)`: one state direction for every action, so the logit gap between two actions never depends on the state ([`tests/test_gru_readout.py`](../tests/test_gru_readout.py)). The same GRU with one readout vector per action (`gru_action_readout`, the readout form the qubit and real-operator models already have) reaches the ceiling with 3,200 sequences: within about 0.001 in every world–seed pair.

## 1. Ceiling decomposition

[`results/mobius-v3-ceiling.json`](../results/mobius-v3-ceiling.json), produced by [`qword/ceiling.py`](../qword/ceiling.py). Mean test NLL over seeds 2026–2028 on exactly the receipt's test splits (`trajectories(world, 80, 24, seed*13+3)`), nats/token, lower is better.

| | Möbius | Harmonic | Hidden drift |
|---|---:|---:|---:|
| Hidden-state oracle (sees the phases; not a learner target) | 0.6435 | 0.6423 | 0.6539 |
| **Bayes ceiling**: true law + prior, actions and past outcomes, 5,000 particles | **0.6740** | **0.6730** | **0.6790** |
| Receipt's 48-particle physics filter, recomputed (identical to the receipt) | 0.6751 | 0.6740 | 0.6809 |
| Open loop: true law + prior pushed through the known actions, outcomes never used | 0.6896 | 0.6889 | 0.6889 |
| Uniform | 0.6931 | 0.6931 | 0.6931 |
| Share of the oracle's lead that is not in the observations | 61% | 60% | 64% |
| Share of the ceiling's gain reached by the open loop | 19% | 21% | 30% |

Particle check, Möbius world: 1,000 / 5,000 / 20,000 particles give 0.6738 / 0.6740 / 0.6739. The Möbius and harmonic worlds are deterministic, so the ceiling uses importance sampling from the prior. Hidden drift uses a bootstrap particle filter that resamples when the effective sample size drops below half. A point-mass prior at the true initial shape reproduces the oracle exactly in the deterministic worlds ([`tests/test_ceiling.py`](../tests/test_ceiling.py)).

The Bayes filter is optimal in expectation. On 80 finite test sequences a learned model can land slightly above or below it.

## 2. Learned models trained to convergence

[`results/mobius-v3-converged.json`](../results/mobius-v3-converged.json), produced by [`qword/converged.py`](../qword/converged.py). 3,000 updates, validation every 100, final weights are the best on validation; learning rates as published (qubit 0.012, others 0.006); minibatch 40; same test splits as above. The 160-sequence regime uses v3's own training and validation splits; the 3,200-sequence regime draws new ones (`seed*13+101`, `seed*13+102`, 400 validation sequences).

**3,200 training sequences**

| Model | Möbius | Harmonic | Hidden drift |
|---|---:|---:|---:|
| Bayes ceiling (from §1) | 0.6740 | 0.6730 | 0.6790 |
| `gru_action_readout` (GRU, one readout vector per action; 151 parameters) | **0.6740** | **0.6727** | **0.6793** |
| `real_operator` (87) | 0.6757 | 0.6742 | 0.6797 |
| `quantum_instrument` (33) | 0.6766 | 0.6765 | 0.6822 |
| `window_transformer` (1,713) | 0.6865 | 0.6840 | 0.6889 |
| `gru`, original readout (144) | 0.6929 | 0.6928 | 0.6933 |
| One-step tabulation | 0.6910 | 0.6907 | 0.6913 |

Head to head across the 9 world–seed pairs: `real_operator` beats `quantum_instrument` in 8; `gru_action_readout` beats `real_operator` in 8 and `quantum_instrument` in 9. Per seed, `gru_action_readout` sits within about 0.001 of the ceiling everywhere (largest gap 0.00101).

**160 training sequences (the published size)**

| Model | Möbius | Harmonic | Hidden drift |
|---|---:|---:|---:|
| `quantum_instrument` | **0.6813** | **0.6807** | 0.6871 |
| `gru_action_readout` | 0.6882 | 0.6871 | **0.6866** |
| `real_operator` | 0.6882 | 0.6885 | 0.6884 |
| `gru`, original readout | 0.6937 | 0.6960 | 0.6946 |
| `window_transformer` | 0.6958 | 0.6960 | 0.7021 |
| One-step tabulation | 0.6926 | 0.6925 | 0.6929 |

At this size `quantum_instrument` beats `real_operator` and `gru_action_readout` in 7 of 9 pairs each. The best-validation steps show why: `real_operator` and the transformer mostly peak at 100–200 updates and then overfit, while in the Möbius and harmonic worlds the 33-parameter qubit model keeps improving to 2,600–2,700 updates in two of three seeds. 160 sequences × 24 tokens at 0.014–0.020 nats/token of learnable signal (the ceiling's gain over uniform) is too little for the larger models.

**Answer to v3's falsifier 1.** Longer training alone does not remove the qubit model's edge; more data does. The edge measured how well 33 constrained parameters resist overfitting, not a geometric advantage on these worlds.

## 3. The same GRU in v2

[`results/sequence-v2-converged.json`](../results/sequence-v2-converged.json): v2's published data sizes (350 / 60 / 110 sequences, length 20), same convergence protocol, minibatch 48.

| Model | Planted qubit source | Planted HMM source |
|---|---:|---:|
| Oracle (source law) | 0.5870 | 0.6591 |
| `quantum_instrument` | **0.5899** | 0.6671 |
| `real_operator` | 0.6027 | **0.6651** |
| `gru_action_readout` | 0.6048 | 0.6652 |
| `window_transformer` | 0.6389 | 0.6749 |
| `gru`, original readout | 0.6501 | 0.6777 |
| One-step tabulation | 0.6245 | 0.6674 |

The fix moves the GRU from worst recurrent model to level with `real_operator`. `quantum_instrument` still wins its own planted generator, as it should: that source is its model class. On the HMM source the learned models barely beat the one-step table at this data size.

## 4. What changes in the v3 conclusions

- "The physics filter wins all three worlds." It is the ceiling. With enough data a small learned model reaches it.
- "The qubit-constrained recurrence beats the other learned models on M/H." True at 160 sequences; reversed at 3,200.
- "Consistent with a geometric inductive bias benefiting a structured generator." Not supported here. The design choice that mattered was letting the action choose the readout.
- The remaining distance to the oracle is missing information, not missing modelling capacity. With random actions, about 60% of what the oracle knows never reaches any observer in 24 steps. Choosing actions that separate competing hypotheses is where headroom remains; such a test should be scored on held-out probes, not on NLL of the chosen actions.

## 5. A note on the two groups (separable from the rest)

v3 describes SU(1,1) oscillator maps and SU(2) qubit rotations as different groups sharing composition and dimension. They are closer. Q-word's measurement update is exactly relativistic velocity addition: before its conditional rotation, `quantum_step` sends a Bloch vector `r` to `(s·η·n) ⊕ r`, Einstein's addition with c = 1 ([`tests/test_sequence.py`](../tests/test_sequence.py), checked to 1e-12). The Bloch ball is relativistic velocity space, the Beltrami–Klein model of hyperbolic space (Kim, arXiv:1109.6564, building on Ungar's gyrovector spaces). Weak measurements act as Lorentz boosts and unitaries as rotations. The oscillator's Möbius maps are the part of the same Lorentz group that acts in one plane. The real difference is when a boost happens. A qubit boosts only as the heralded result of a measurement, because an unconditional boost can increase the trace distance between two states, which no quantum channel can do. The oscillator pings boost unconditionally.

## Limits

- Three seeds, 80 test sequences per seed (v2: 110). Differences of about 0.001 are near test noise; the head-to-head counts above are the more reliable summary.
- No hyperparameter search. Learning rates, widths and minibatch sizes are the published ones; `gru_action_readout` changes one design choice and is not tuned.
- Equal updates are not equal parameters, FLOPs, memory or energy.
- The ceiling uses the true law and prior. It is the best possible observation-only predictor, not a learner.
- Classical simulation throughout; no quantum hardware and no claim of quantum advantage.

## Reproduce

```bash
pip install -e '.[sequence]'
python -m unittest discover -s tests -v
python -m qword.ceiling --output results/mobius-v3-ceiling.json            # NumPy only, ~10 min
python -m qword.converged --suite v3 --regimes published large --output results/mobius-v3-converged.json
python -m qword.converged --suite v2 --regimes published --output results/sequence-v2-converged.json
```

The v3 training runs can be split by world (`--items mobius`) and combined with `--merge part1.json part2.json ... --output results/mobius-v3-converged.json`. The committed receipts were produced that way on CPU with Python 3.13, NumPy 2.5 and PyTorch 2.14, one thread per process.
