# Q-word v2: learnable predictive state and a transformer baseline

**Date:** 2026-10-08. **Status:** exploratory synthetic study; not preregistered before inspecting outcomes. The original v1 scientific gates remain distinct.

## Question

Does enforcing **qubit-channel geometry** make a small recurrent predictor easier to learn than a comparably small, unconstrained classical predictor? Compare actual held-out future predictions, not an analogy between matrix symbols.

## Controlled data

Two environments produce sequences of `a_t in {0,1,2}` and `y_t in {0,1}`. The model receives current action and **only previous** observations when predicting `Pr(y_t=1 | a_0..a_t,y_0..y_{t-1})`.

1. **Planted qubit instrument.** Bloch state `r` in the unit ball; each action selects a POVM measurement axis and sharpness. Outcome `y` yields `P(y|r,a)=(1+s_y eta_a n_a·r)/2`, a Lüders posterior, and a conditional noncommuting `SU(2)` rotation. The generator is numerical classical simulation of a qubit; its exact parameters are never supplied to training.
2. **Planted classical HMM.** Three-state controlled belief filter; each action specifies state-dependent Bernoulli emission probabilities and a distinct transition matrix. The oracle carries the exact Bayesian belief distribution.

Train, validation and test contain separate seeded trajectories. For every token the generator retains its exact **oracle next-observation probability**, allowing both held-out log loss and prediction error relative to the source law. The oracle NLL on finite sampled test sequences is not a theoretical minimum for every finite sample.

## Four learned models

| Architecture | Trainable parameters | Persistent state (real coordinates) | Remarks |
|---|---:|---:|---|
| Quantum-constrained instrument | 33 | 3 | Positive Bloch state, valid unsharp effects, conditional norm-preserving rotations; simulated entirely on a classical CPU |
| Real-operator recurrence | 87 | 3 | Learned action/outcome matrices with `tanh`; a **nonlinear** real-state baseline, not an exact linear PSR |
| GRU | 144 | 4 | Conventional recurrent gating and readout |
| Causal window transformer | 1,713 | Last 8 positions at width 8 | One encoder layer, two heads, masking of future tokens and older tokens; no weight/pretraining advantage |

Also evaluate uniform `p=0.5`, one-step action/previous-action/previous-symbol tabulation with smoothing, and the exact generator oracle. **Equal optimization steps do not imply equal parameters, wall-clock, inference FLOPs, physical storage, or energy.** These numbers are not a fair physical-memory comparison. Parameter counts exclude optimizer state.

## Fixed *for this exploratory run* (not pre-registered)

- Three independent root seeds: 2026, 2027, 2028, giving disjoint train/validation/test trajectories within each seed.
- Each task/seed: 350 training, 60 validation, 110 test sequences; 20 tokens each, 80 Adam steps per model, minibatch 48.
- Learning rate `0.012` for qubit instrument and `0.006` for each other model; training losses use supervised teacher forcing; no early stopping or post-test hyperparameter selection.
- Main score: test Bernoulli negative log-likelihood in **nats/token** (lower better). Secondary: mean absolute error from source oracle and later-half NLL.
- CPU PyTorch. Test batch probabilities are predicted from previous observed outcomes; current and future outcomes never enter predictions for their own timesteps. No teacher-parameter initialization, and random model initializations vary by root seed.

## Recorded held-out results

Mean over the three seeds (small sample; **no significance claim**):

| Model | Qubit source NLL | HMM source NLL |
|---|---:|---:|
| Oracle | 0.58702 | 0.65915 |
| One-step tabulation | 0.62450 | 0.66744 |
| Quantum-constrained | **0.61336** | 0.67593 |
| Real-operator | 0.64146 | **0.66764** |
| GRU | 0.65776 | 0.67851 |
| Window transformer | 0.64187 | 0.68188 |

The quantum-constrained model has the lowest learned-model NLL on the planted qubit source in seeds **2026 and 2028**, but not 2027, where the real-operator recurrence does slightly better (0.63596 versus 0.63880). On the classical HMM source the real operator beats the quantum-constrained model in **all three** seeds. Even on the qubit source, the exact oracle is better than every learned model.

This is **not** evidence of quantum speed-up, physical quantum memory savings, superior LLM inference, or generalization to real datasets. The favorable source was created from the same instrument class as one contender, making the study an architectural inductive-bias check, not evidence of novelty. The classical model is free to simulate the qubit state, and the quantum-constrained model is classically implemented using three real coordinates. The transformer has a deliberately short eight-token window and no pretrained weights, so a production-transformer comparison is **not** claimed.

## Important falsifiers and next independent round

- Increase training to convergence and tune each model on validation only, including parameter/compute-matched recurrent and state-space models. The present ranking may reflect optimization speed or learning-rate choice.
- Replace the current easy fixed generators with previously unseen families; randomize qubit axes, strengths, and conditional rotations across tasks. The current model may be adapted to only one hand-built process.
- Add robust multi-seed confidence intervals and larger held-out sets. Three seeds are insufficient for discovery claims.
- Measure latency, actual allocated bytes, energy and every state/read/reset cost if making a hardware resource claim.
- Test longer observation gaps, noisy inputs, distribution drift and partial conditioning. Maintain causal masking and the same observation rights.

## Run

```bash
pip install -e '.[sequence]'
python -m unittest discover -s tests -v
python -m qword.sequence --task both --train 350 --valid 60 --test 110 --length 20 --steps 80 --seed 2026 --output results/sequence-v2-seed2026.json
```

Repeat with `--seed 2027` and `--seed 2028`. Code is in `qword/sequence.py`. Summary receipt and per-seed JSON are in `results/`; the interactive comparison is [`web/sequence.html`](../web/sequence.html).
