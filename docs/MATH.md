# The mathematics of Q-word

## Predictive states: rows are histories, columns are future tests

`D_{ht}=Pr(t|h)` for any history `h` and test `t`. Finite-dimensional linear predictive representations factor a rank-`k` table as `D=B M`, where rows of `B` are predictive states and columns of `M` are test readers. `B→BS`, `M→S^{-1}M` preserves all predictions for any invertible `S`: coordinate freedom does not imply physical neural drift, and it does not prove the listener remains invariant if the reader is held fixed.

Quantum instruments supply a constrained physical factorization `D_{ht}=Tr(rho_h E_t)` where `rho_h` is positive semidefinite and has unit trace, and `E_t` is a valid positive effect. For sequential tests the effect is constructed by composing the instrument's dual maps. A `d`-dimensional quantum density matrix lives in the real vector space of Hermitian `d×d` matrices (dimension `d²`), so the rank of a prediction table represented by such an instrument is at most `d²`. The reverse implication need not hold: low rank does not guarantee valid small quantum states/effects.

## Trine quantum memory

Let `n_i` be three unit vectors on the Bloch equator at angles `2πi/3`, with `n_i·n_j=1` if `i=j`, otherwise `-1/2`. With Pauli matrices `σ=(σx,σy,σz)`, define `rho_i=(I+n_i·σ)/2`, `E_j=(I-n_j·σ)/3`.

- `rho_i` has eigenvalues `(1,0)` and trace 1.
- `E_j` has eigenvalues `(2/3,0)` and is positive semidefinite.
- `Σ_j E_j=I`, since `Σ n_j=0`.
- `Tr((a·σ)(b·σ))=2a·b`.

Therefore `Pr(j|i)=Tr(rho_i E_j)=(1-n_i·n_j)/3`, giving `P=(J-I)/2`. As an *instrument*, outcome `j` maps any input `rho` to `Tr(E_j rho) rho_j` (measure and reprepare). This is a completely positive trace-nonincreasing map; the summed channel preserves trace. A chain that starts in `rho_i` thus exactly reproduces all no-repeat sequence probabilities. The quantum memory is a qubit; the three-outcome measurement and conditional preparation require physical control hardware. A persistent output-conditioned classical state inside the device would change the memory accounting.

`P` has eigenvector `(1,1,1)` with eigenvalue 1 and two perpendicular eigenvectors with eigenvalue `-1/2`. The singular values are `(1,1/2,1/2)`. The best unconstrained rank-2 approximation has Frobenius error 0.5; nonnegative stochastic rank-2 factorizations cannot do better. That is only a **one-step** lower bound over three conditional histories, not an optimization result for every classical architecture.

## Oscillator observable → qubit drive

For phases `theta_j`, define `m_l=(1/N) Σ_j exp(i l theta_j)`. The sample populations A and B have equal `m1`, so every probe restricted to a deterministic function of `m1` alone receives the exact same input. Feeding `m2` permits distinction; the quantum equations are not responsible for the new input information.

A qubit Hamiltonian `H=(ℏ/2) h·σ`, with `h=(Re(m1+g2m2),Im(m1+g2m2),omega)`, evolves as `U=cos(|h|t/2)I -i sin(|h|t/2) (h·σ)/|h|`. From `rho0=(I+σy)/2`, `p0=Tr(|0><0| U rho0 U†)`. Equivalently a classical Bloch vector `r0=(0,1,0)` rotates around `n=h/|h|` by `|h|t`, `r'=r cos(a)+(n×r) sin(a)+n(n·r)(1-cos(a))`, and `p0=(1+r'_z)/2`. This demonstrates a complete classical simulation with the same information and accuracy for a single qubit under classical drive.

## What a ping changes, and what a measurement loses

Conjugation by a unitary is reversible. An unread measurement in computational basis maps `rho→P0 rho P0 + P1 rho P1` and removes off-diagonal coherence. For initial `|+x>`, purity falls from `1` to `1/2` after dephasing. Unitary conjugation preserves density-matrix eigenvalues and purity, so no subsequent unitary on the qubit alone restores the unknown original pure state from this maximally mixed reduced state.

A group commutator `Ux Uy Ux† Uy†` is nontrivial for generic rotations around x/y; its leading correction is order `epsilon²`, related to `[σx,σy]=2iσz`. Noncommuting SU(2) rotations must not be equated to direct measurement of Berry phase; that needs a suitable closed path and phase-sensitive reference. Classical SO(3) rotations also do not commute.
