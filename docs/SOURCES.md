# Literature and provenance

The project has three distinct sources of inspiration and should not be represented as discovering their established mathematics.

1. **Predictive state representations:** Singh, James, Rudary (2004), *Predictive State Representations: A New Theory for Modeling Dynamical Systems*, UAI, [arXiv:1207.4167](https://arxiv.org/abs/1207.4167). Their `D` matrix is past-conditional future-test predictions; note it is not a Hamiltonian or a density matrix.
2. **Quantum memory of stochastic processes:** Gu et al. (2012), *Quantum mechanics can reduce the complexity of classical models*, Nature Communications 3, 762. [DOI](https://doi.org/10.1038/ncomms1761). Follow-up quantum stochastic-simulator literature treats nonorthogonal quantum causal memories and physically constrained memory comparisons.
3. **Experimental quantum memory advantage:** Ghafari et al. (2019), *Dimensional quantum memory advantage in the simulation of stochastic processes*, [arXiv:1812.04251](https://arxiv.org/abs/1812.04251). Physical quantum-memory examples predate this repository.
4. **Matrix methods:** Eckart & Young (1936) / Mirsky (1960) low-rank approximation; standard Pauli-matrix / Bloch-sphere / POVM quantum information machinery.
5. **Oscillator connection:** [MovingTarget2](https://github.com/anttiluode/MovingTarget2) proves that first moments predict weak present readouts but need not close later phase updates. Q-word's first/second-moment pair is an intentionally transparent illustrative counterexample, not newly measured neurons.
6. **Read and disturbance:** [VMN](https://github.com/anttiluode/VMN) motivates testing repeated interactions; the Q-word dephasing test is textbook quantum information, not a reproduction of VMN's physical mechanism.

**Scope of citation:** Q-word has no independent quantum hardware data or real brain recordings. Equivalence of the qubit and classical observable process is checked mathematically and numerically. The oscillator-to-qubit coupling is an engineered classical control channel, not entanglement or a quantum biological model.
