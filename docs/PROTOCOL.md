# Q-word v1 — registered questions and failure conditions

Status: prospective *design of the gates*. The formulas are established; this project does not claim a new physical advantage. In the first phase all quantum mechanics is simulated on a conventional CPU.

## Gate A — exactly generated futures, not a quantum-advantage claim

**Question.** Can the three-symbol no-repeat chain be realized both by a conventional three-state simulator and an exact one-qubit *instrument*, while a two-dimensional linear-state factorization is ruled out for its one-step prediction table?

- Target table: `P = (ones(3,3)-I)/2`, conditionally on the previous observed symbol.
- Quantum states: `rho_i=(I+n_i·sigma)/2` for trine equatorial directions `n_i`.
- Quantum effects: `E_j=(I-n_j·sigma)/3`, and on emitting `j` the instrument prepares `rho_j`.
- **Pass** exact: `max_abs(Tr(rho_i E_j)-P_ij) <= 1e-12`; POVM sums to I, all states/effects PSD; singular rank(P)=3; rank-2 Frobenius lower bound sigma_3(P)=0.5.
- **Pass** sampling diagnostic (not evidence of quantum advantage): with seed=4100 and 12,000 transitions, both empirical tables within 0.04 max absolute deviation of target P. Single stream deterministic given seed.
- **Fail** if probabilities, normalization, PSD, rank arithmetic or sampling diagnostic fail.

**Memory accounting.** A classical exact causal-state simulator has three distinguishable persistent states; the qubit instrument has Hilbert dimension two and uses three nonorthogonal density states. These are *different physical encoding classes* and the comparison is about Hilbert dimension, not gate complexity, full machine memory, classical finite-precision arithmetic, energy, total cost or proven hardware advantage. Three-outcome POVM, preparation feedback, emitted symbols, and external recorder are not free. No-repeat-chain probability matrix rank3 proves that no two-state **linear** predictive representation captures all its one-step histories exactly; it is not a universal lower bound on arbitrary programs.

## Gate B — oscillator-to-qubit observation and a required classical control

**Question.** Can two oscillator populations that agree at `m1` but differ at `m2` become distinguishable when a probe couples to the second harmonic?

- Constellations `A=[pi/3,-pi/3,pi/3,-pi/3]`, `B=[0,0,0,pi]`; `m1(A)=m1(B)=0.5`, `m2(A)=-0.5`, `m2(B)=1`.
- Hamiltonian `H/hbar = (h_x sigma_x + h_y sigma_y + 0.4 sigma_z)/2`, with `h_x+i h_y = m1+g2*m2` and initial probe `rho=(I+sigma_y)/2`.
- Probe duration `dt=1.1`, with `g2=0` vs `g2=1`.
- **Pass** equal first-only probabilities (<1e-12 gap), second-harmonic gap >0.05, exact unitary and density positivity.
- **Mandatory negative control** classical Bloch-vector rotation with identical moments reproduces all qubit output probabilities within `1e-12`.
- **Fail** if claimed 'quantum advantage' is based on this toy gap, since it is a better classical *input signal* (m2), not proof of quantum processing superiority.

## Gate C — operations versus observations

- Distinguish reversible unitary evolution from unread dephasing, which loses reduced-state purity and cannot be undone by an unknown-state-independent unitary on the measured system alone.
- Verify SU(2) noncommutative group commutator nonidentity for nonzero angle.
- Do not conflate this with a Berry-phase detection, a classical Möbius group's geometry, quantum neural activity or a physical realization.

## Independent challenge beyond v1

To claim a *new* advantage over existing science, preregister unseen controlled processes with accessible interfaces, matched physical memory accounting, same observation channel and budget, matched learned classical POMDP/PSR/state-space baselines, and report log loss, wall-clock and energy if actually measured. Include no-advantage cases. In particular, **a solitary qubit under a classical drive is equivalently simulated by a classical Bloch vector** and cannot demonstrate computational quantum advantage by itself.

## Reproduction

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m qword --steps 12000 --seed 4100 --output results/receipt.json
```

The CLI's empirical transition errors depend on finite sampling. The mathematical identities do not. The HTML demo is a separate in-browser derivation/visualization; its deterministic random path is *not* the Python NumPy RNG path.
