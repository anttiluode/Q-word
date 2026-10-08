import math
import unittest
import numpy as np
from qword.core import (
    I2, SX, SY, SZ, circular_moment, trine_instrument, no_repeat_matrix,
    quantum_transition_matrix, classical_rank_two_lower_bound,
    simulate_symbols, empirical_transition_matrix, stationary_distribution,
    sequence_probability, oscillator_constellations, qubit_rotation,
    hybrid_p0, classical_same_information_probe, quantum_dephase_z,
    initial_plus_y, purity, commutator_path, run_receipt,
)


class TestQuantumInstrument(unittest.TestCase):
    def test_valid_states_and_povm(self):
        states, effects = trine_instrument()
        self.assertTrue(np.allclose(sum(effects), I2, atol=1e-13))
        for rho in states:
            self.assertTrue(np.allclose(rho, rho.conj().T))
            self.assertAlmostEqual(np.trace(rho).real, 1)
            self.assertGreaterEqual(np.linalg.eigvalsh(rho).min(), -1e-12)
        for e in effects:
            self.assertGreaterEqual(np.linalg.eigvalsh(e).min(), -1e-12)

    def test_exact_no_repeat(self):
        self.assertTrue(np.allclose(quantum_transition_matrix(), no_repeat_matrix(), atol=1e-14))
        self.assertEqual(np.linalg.matrix_rank(no_repeat_matrix()), 3)
        self.assertAlmostEqual(classical_rank_two_lower_bound(), 0.5)

    def test_stationary_and_sequence(self):
        self.assertTrue(np.allclose(stationary_distribution(no_repeat_matrix()), [1/3]*3))
        self.assertAlmostEqual(sequence_probability(np.array([0,1,0,2])), 0.125)
        self.assertEqual(sequence_probability(np.array([0,0])), 0)

    def test_reproducibility(self):
        for model in ("classical", "quantum"):
            a = simulate_symbols(2000, 42, model)
            b = simulate_symbols(2000, 42, model)
            self.assertTrue(np.array_equal(a,b))
            self.assertTrue(np.all(a[:-1] != a[1:]))
            self.assertLess(np.max(abs(empirical_transition_matrix(a)-no_repeat_matrix())), 0.09)
        self.assertTrue(np.array_equal(simulate_symbols(2000,42,"classical"), simulate_symbols(2000,42,"quantum")))

    def test_bad_inputs(self):
        with self.assertRaises(ValueError): simulate_symbols(-1)
        with self.assertRaises(ValueError): simulate_symbols(1, model="not-a-model")
        with self.assertRaises(ValueError): circular_moment(np.array([0]), harmonic=0)


class TestOscillatorProbe(unittest.TestCase):
    def test_memory_alias_and_distinct_hidden_moment(self):
        a,b=oscillator_constellations()
        self.assertAlmostEqual(circular_moment(a).real,0.5)
        self.assertAlmostEqual(circular_moment(b).real,0.5)
        self.assertAlmostEqual(circular_moment(a,2).real,-0.5)
        self.assertAlmostEqual(circular_moment(b,2).real,1.0)
        self.assertLess(abs(hybrid_p0(a,g2=0)-hybrid_p0(b,g2=0)),1e-12)
        self.assertGreater(abs(hybrid_p0(a,g2=1)-hybrid_p0(b,g2=1)),0.05)

    def test_no_quantum_advantage_in_first_prototype(self):
        a,b=oscillator_constellations()
        for phases in (a,b):
            for g1,g2,omega,dt in ((1,0,.4,1.1),(.2,1.4,.7,.7),(1,1,.4,1.1)):
                self.assertAlmostEqual(hybrid_p0(phases,g1=g1,g2=g2,omega=omega,dt=dt),
                   classical_same_information_probe(phases,g1=g1,g2=g2,omega=omega,dt=dt),places=12)

    def test_unitarity_and_purity(self):
        for h in ([0,0,0],[.5,-.2,.7]):
            u=qubit_rotation(np.array(h),.8)
            self.assertTrue(np.allclose(u.conj().T@u,I2,atol=1e-13))
            self.assertAlmostEqual(purity(u@initial_plus_y()@u.conj().T),1.0)

    def test_measurement_drops_coherence(self):
        rho=(I2+SX)/2
        dephased=quantum_dephase_z(rho)
        self.assertAlmostEqual(purity(rho),1)
        self.assertAlmostEqual(purity(dephased),0.5)
        self.assertTrue(np.allclose(dephased,I2/2))
        # Unitaries preserve eigenvalues; they cannot restore unknown coherence here.

    def test_noncommuting_path(self):
        self.assertTrue(np.allclose(commutator_path(0),I2))
        self.assertFalse(np.allclose(commutator_path(.3),I2,atol=1e-5))
        self.assertTrue(np.allclose(commutator_path(.3).conj().T@commutator_path(.3),I2,atol=1e-12))

    def test_receipt_is_numeric_and_honest(self):
        record=run_receipt(1000,33)
        self.assertLess(record["quantum_exact_max_error"],1e-12)
        self.assertAlmostEqual(record["classical_rank_two_frobenius_lower_bound"],.5)
        self.assertGreater(record["oscillator"]["first_second_gap"],.05)
        self.assertLess(record["oscillator"]["quantum_vs_classical_bloch_max_gap"],1e-12)

if __name__=="__main__": unittest.main()
