# Q-word — when memory is a prediction, not a snapshot

**A runnable laboratory for the difference between storing a hidden state and preserving its future responses.** A conventional classical Markov chain, a qubit predictive-state instrument, and an oscillator-driven qubit probe all live in one tiny NumPy codebase. The interactive visual laboratory is [web/index.html!](https://anttiluode.github.io/Q-word/web/index.html) .

> This repo **does not** demonstrate a novel quantum advantage, quantum neurons, quantum consciousness, or evidence of quantum biology. It numerically reproduces one known quantum-memory compression construction, demonstrates a classical/quantum measurement interface, and carries the negative controls that prevent attractive visual analogies from masquerading as physical discoveries.

## 2026-10-08 — Möbius transfer: independently specified oscillator dynamics (v3)

[Interactive Möbius transfer viewer!](https://anttiluode.github.io/Q-word/web/mobius.html) · [full protocol and limitations](docs/MOBIUS_TRANSFER.md) · [raw three-seed receipt](results/mobius-v3.json). This adds three **classical** hidden-oscillator environments: exact reversible first-harmonic reads, shape-writing second-harmonic dynamics, and unknown per-oscillator drift. Four original learned models get only actions and past sampled observations; an additional law-informed particle filter gets **the same observations** but knows the correct equations and the prior over hidden shapes. The privileged full-state oracle is kept separate.

Mean held-out NLL (lower better): physics filter 0.6751 / 0.6740 / 0.6809, qubit-inspired learner 0.6824 / 0.6816 / 0.6892, real-operator learner 0.6879 / 0.6848 / 0.6872, across exact Möbius / second harmonic / hidden drift. **The physics filter wins all three worlds; no quantum advantage is shown.** A separate numeric invariant check confirms exact known-read composition/inversion to near floating-point error but a blind inverse fails with unknown drift. Model rankings are exploratory, with only three seeds and a short, nonmatched training budget.

## Three experiments

| Experiment | What it actually establishes | What it does not |
|---|---|---|
| **A. Three symbols, one qubit** | An exact three-outcome qubit measure-and-prepare instrument generates the three-state no-repeat chain; rank of the one-step prediction matrix is 3 | A hardware speed-up, total machine-memory saving, or a universal bound on arbitrary classical software |
| **B. Oscillator → qubit probe** | Two oscillator arrangements have identical first phase moments but different second moments; second-harmonic driving makes their qubit output distinguishable | Quantum advantage: a matched *classical Bloch vector* generates exactly the same numbers |
| **C. Noncommuting pings** | SU(2) group commutator can encode the order of four rotations; unread projective measurement loses local coherence | Berry-phase interferometry, a neural memory mechanism, or a quantum-brain model |

## The predictive matrix and the quantum factorization

A system's observable past/future matrix is `D[history, future_test] = Pr(future_test | history)` (Singh, James & Rudary, UAI 2004). Its rank measures linear predictive dimension, *not* classical hardware memory cost by itself.

For a three-symbol process that can never repeat its latest symbol:

```text
         next A  next B  next C
last A      0      1/2     1/2
last B     1/2      0      1/2
last C     1/2     1/2      0
```

This table has singular values `1, 0.5, 0.5`, so any rank-2 approximation to the **one-step** table has Frobenius error at least `0.5` (Eckart–Young). An exact 3-causal-state classical simulator exists.

Take three equatorial Bloch directions `n_i = (cos(2πi/3), sin(2πi/3), 0)`. Use a single qubit with `rho_i=(I+n_i·σ)/2` and POVM effects `E_j=(I-n_j·σ)/3`. Then

```text
Pr(j | i) = Tr(rho_i E_j) = (1 - n_i·n_j)/3
```

is `0` for `i=j` and `1/2` otherwise. Emit `j`, then reprepare `rho_j`. **The three output histories correspond to three nonorthogonal qubit memories in a two-dimensional Hilbert space.** The process statistics agree exactly; physical control and measurement overhead remain outside this one-qubit memory comparison. This is a small example of existing quantum stochastic-simulation research, not a new quantum compression theorem.

## The oscillator memory experiment

The two chosen populations of four phases are:

```text
A: +60°, -60°, +60°, -60°
B:   0°,   0°,   0°, 180°
```

With `m_k = mean(exp(i k theta))`, they both have `m1=0.5`, but `m2(A)=-0.5`, `m2(B)=1`.

Feed their classical moments into a *numerically simulated* qubit Hamiltonian:

```text
H/ℏ = [Re(m1 + g2*m2) σx + Im(m1 + g2*m2) σy + 0.4 σz] / 2
```

From an initial `|+y>` state, measure `Pr(|0>)` after `dt=1.1`. With `g2=0`, the two responses coincide. With `g2=1`, they differ substantially. However, an ordinary classical Rodrigues rotation of a three-dimensional Bloch vector produces the **identical** probability: this is an observation-resolution effect, *not* a quantum speed-up or memory benefit.

## Try it

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m qword --steps 12000 --seed 4100 --output results/receipt.json
```

The standard library + NumPy are enough for the research code. `web/index.html` works offline and needs no build system. The Python CLI exports an inspectable JSON measurement receipt. The [protocol](docs/PROTOCOL.md) freezes gates, falsifiers and accounting limitations; the [mathematics note](docs/MATH.md) derives each step; [sources](docs/SOURCES.md) distinguish literature from our numerical examples.

## How this relates to the other projects

- [MovingTarget2](https://github.com/anttiluode/MovingTarget2): a preserved weak-response code is not necessarily sufficient for later state updates. Its first-moment/second-moment problem supplies our oscillator pair.
- [AnttisBrain2](https://github.com/anttiluode/AnttisBrain2): moving moons supplied the visual language; they are not a quantum simulator.
- [BrainLoops](https://github.com/anttiluode/BrainLoops): measurable EEG recurrence supplies a separate empirical question, not evidence for a quantum mechanism.
- [VMN](https://github.com/anttiluode/VMN): interaction/read disturbance motivates the unitary-versus-measurement separation.

**The research question left open:** can a restricted quantum instrument preserve a process's future statistics more efficiently than a strong classical predictor after **the full cost of measurement, memory, preparation, resets, and read disturbance** has been accounted for? This v1 makes that question precise; it doesn't claim to solve it.
