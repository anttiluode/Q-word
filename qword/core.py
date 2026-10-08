"""Q-word: finite-dimensional quantum instruments and classical oscillator probes.

All quantum dynamics are *classically simulated*, never hardware measurements.
The no-repeat model is a measure-and-prepare qubit instrument, not a unitary-only
qubit simulator. Effects E_j form a three-outcome POVM, and instrument outcome j
resets the internal qubit memory to rho_j.
"""
from __future__ import annotations

import numpy as np

I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = np.array([SX, SY, SZ])


def trine_directions() -> np.ndarray:
    """Three unit vectors 120 degrees apart on a Bloch-sphere equator."""
    phase = 2 * np.pi * np.arange(3) / 3
    return np.stack([np.cos(phase), np.sin(phase), np.zeros(3)], axis=1)


def bloch_operator(vector: np.ndarray) -> np.ndarray:
    """v · sigma; Hermitian for any real three-vector v."""
    v = np.asarray(vector, dtype=float)
    if v.shape != (3,):
        raise ValueError("expected three real coordinates")
    return np.einsum("i,ijk->jk", v, PAULI)


def trine_instrument() -> tuple[np.ndarray, np.ndarray]:
    """Returns rho_i=(I+n_i.sigma)/2 and E_j=(I-n_j.sigma)/3."""
    axes = trine_directions()
    states = np.stack([(I2 + bloch_operator(n)) / 2 for n in axes])
    effects = np.stack([(I2 - bloch_operator(n)) / 3 for n in axes])
    return states, effects


def no_repeat_matrix() -> np.ndarray:
    """The exact observed transition law; state equals last emitted symbol."""
    return (np.ones((3, 3), dtype=float) - np.eye(3)) / 2


def quantum_transition_matrix() -> np.ndarray:
    states, effects = trine_instrument()
    values = np.einsum("iab,jba->ij", states, effects).real
    return values


def classical_rank_two_lower_bound() -> float:
    """Best possible Frobenius error of any rank <= 2 approximation to P.

    Stochastic 2-latent-state factorization is a subset of rank <= 2, so the
    singular-value lower bound also applies to those classical predictors.
    This bound is for the three one-step conditional predictions P, not a
    fitted 2-state HMM likelihood bound on arbitrary sequences.
    """
    return float(np.linalg.svd(no_repeat_matrix(), compute_uv=False)[-1])


def stationary_distribution(matrix: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eig(np.asarray(matrix).T)
    idx = np.argmin(abs(vals - 1))
    v = np.asarray(vecs[:, idx].real)
    v = v / v.sum()
    return v


def simulate_symbols(steps: int, seed: int = 2026, model: str = "quantum", start: int = 0) -> np.ndarray:
    """Sample observable emissions. The 'quantum' instrument resets to rho_j.

    Both physical models produce the same conditional law; matching sampled
    symbols for equal RNG streams does *not* imply hardware-level equivalence.
    """
    if steps < 0 or start not in (0, 1, 2):
        raise ValueError("steps >=0, start in {0,1,2} required")
    if model not in ("quantum", "classical"):
        raise ValueError("model must be quantum or classical")
    rng = np.random.default_rng(seed)
    transition = quantum_transition_matrix() if model == "quantum" else no_repeat_matrix()
    out = np.empty(steps + 1, dtype=int)
    out[0] = start
    for k in range(steps):
        current = out[k]
        probabilities = transition[current].copy()
        probabilities = np.maximum(probabilities, 0)
        probabilities /= probabilities.sum()
        out[k + 1] = rng.choice(3, p=probabilities)
    return out


def empirical_transition_matrix(symbols: np.ndarray) -> np.ndarray:
    counts = np.zeros((3, 3), dtype=int)
    for i, j in zip(symbols[:-1], symbols[1:]):
        counts[i, j] += 1
    denominators = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, denominators, out=np.zeros((3, 3)), where=denominators != 0)


def sequence_probability(symbols: np.ndarray, matrix: np.ndarray | None = None) -> float:
    """Conditional probability given the first symbol (not a joint likelihood)."""
    if matrix is None:
        matrix = no_repeat_matrix()
    if len(symbols) < 2:
        return 1.0
    return float(np.prod([matrix[int(i), int(j)] for i, j in zip(symbols[:-1], symbols[1:])]))


def oscillator_constellations() -> tuple[np.ndarray, np.ndarray]:
    """Two four-phase memories whose first moments coincide but second do not."""
    a = np.array([np.pi / 3, -np.pi / 3, np.pi / 3, -np.pi / 3])
    b = np.array([0.0, 0.0, 0.0, np.pi])
    return a, b


def circular_moment(phases: np.ndarray, harmonic: int = 1) -> complex:
    if harmonic < 1:
        raise ValueError("harmonic must be >=1")
    return complex(np.mean(np.exp(1j * harmonic * np.asarray(phases))))


def qubit_rotation(hamiltonian_xyz: np.ndarray, dt: float) -> np.ndarray:
    """U=exp(-i dt/2 h.sigma), using the exact Pauli exponential."""
    h = np.asarray(hamiltonian_xyz, dtype=float)
    norm = float(np.linalg.norm(h))
    if norm == 0:
        return I2.copy()
    angle = norm * dt / 2
    return np.cos(angle) * I2 - 1j * np.sin(angle) * bloch_operator(h / norm)


def initial_plus_y() -> np.ndarray:
    return (I2 + SY) / 2


def hybrid_hamiltonian(phases: np.ndarray, *, g1: float = 1.0, g2: float = 0.0, omega: float = 0.4) -> np.ndarray:
    """Classical first/second moment drives a *simulated* qubit Hamiltonian.

    h_xy = g1*m1 + g2*m2, h_z=omega.  No entanglement exists between
    the classical population and the qubit in this model.
    """
    signal = g1 * circular_moment(phases, 1) + g2 * circular_moment(phases, 2)
    return np.array([signal.real, signal.imag, omega])


def hybrid_p0(phases: np.ndarray, *, g1: float = 1.0, g2: float = 0.0, omega: float = 0.4, dt: float = 1.1) -> float:
    """Probability of outcome |0> after a controlled unitary on |+y>."""
    u = qubit_rotation(hybrid_hamiltonian(phases, g1=g1, g2=g2, omega=omega), dt)
    rho = u @ initial_plus_y() @ u.conj().T
    return float(np.clip(rho[0, 0].real, 0, 1))


def classical_same_information_probe(phases: np.ndarray, *, g1: float = 1.0, g2: float = 0.0, omega: float = 0.4, dt: float = 1.1) -> float:
    """Exact classical Bloch-vector evolution: a negative quantum-advantage control.

    The chosen Hamiltonian drives a single qubit without noise/entanglement;
    Rodrigues' rotation reproduces its measurement probability using reals.
    """
    h = hybrid_hamiltonian(phases, g1=g1, g2=g2, omega=omega)
    speed = np.linalg.norm(h)
    r = np.array([0.0, 1.0, 0.0])
    if speed:
        n = h / speed
        theta = dt * speed
        r = r * np.cos(theta) + np.cross(n, r) * np.sin(theta) + n * np.dot(n, r) * (1 - np.cos(theta))
    return float(np.clip((1 + r[2]) / 2, 0, 1))


def quantum_dephase_z(rho: np.ndarray) -> np.ndarray:
    """Unread projective Z measurement, discarding its outcome."""
    return np.diag(np.diag(np.asarray(rho))).astype(complex)


def purity(rho: np.ndarray) -> float:
    return float(np.trace(rho @ rho).real)


def commutator_path(angle: float) -> np.ndarray:
    """SU(2) group commutator Ux Uy Ux^-1 Uy^-1."""
    ux = qubit_rotation(np.array([1., 0., 0.]), angle)
    uy = qubit_rotation(np.array([0., 1., 0.]), angle)
    return ux @ uy @ ux.conj().T @ uy.conj().T


def run_receipt(steps: int = 12000, seed: int = 4100) -> dict:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    theoretical = no_repeat_matrix()
    quantum = quantum_transition_matrix()
    classical_path = simulate_symbols(steps, seed, "classical")
    quantum_path = simulate_symbols(steps, seed, "quantum")
    a, b = oscillator_constellations()
    single = [hybrid_p0(x, g2=0) for x in (a, b)]
    two = [hybrid_p0(x, g2=1) for x in (a, b)]
    classical_two = [classical_same_information_probe(x, g2=1) for x in (a, b)]
    return {
        "schema": "q-word.v1",
        "seed": int(seed),
        "steps": int(steps),
        "models": {"classical_hidden_states": 3, "quantum_memory_dimension": 2, "quantum_memory_qubits": 1},
        "transition_matrix": theoretical.tolist(),
        "quantum_transition_matrix": quantum.tolist(),
        "quantum_exact_max_error": float(np.max(abs(quantum - theoretical))),
        "classical_rank_two_frobenius_lower_bound": classical_rank_two_lower_bound(),
        "classical_empirical_max_error": float(np.max(abs(empirical_transition_matrix(classical_path) - theoretical))),
        "quantum_empirical_max_error": float(np.max(abs(empirical_transition_matrix(quantum_path) - theoretical))),
        "sample_paths_identical_given_seed": bool(np.array_equal(classical_path, quantum_path)),
        "oscillator": {
            "first_moments": [complex(circular_moment(x)).real for x in (a, b)],
            "second_moments": [complex(circular_moment(x,2)).real for x in (a, b)],
            "qubit_first_only_p0": single,
            "qubit_first_second_p0": two,
            "classical_same_information_p0": classical_two,
            "first_only_gap": abs(single[0] - single[1]),
            "first_second_gap": abs(two[0] - two[1]),
            "quantum_vs_classical_bloch_max_gap": float(max(abs(x-y) for x,y in zip(two, classical_two))),
        },
        "scope": "Classical numerical simulation of a qubit instrument and driven unitary. No hardware, no biological data, no quantum advantage inferred from oscillator sensing.",
    }
