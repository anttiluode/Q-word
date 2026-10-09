"""What the v3 observation channel can support: oracle, Bayes ceiling, open loop.

For each Möbius-transfer world and seed this module scores, on the SAME test
trajectories as results/mobius-v3.json:

* the hidden-state oracle (sees the phases; not reachable by any learner);
* the Bayes filter: true law + disclosed prior over initial shapes, actions and
  past outcomes only. Importance sampling from the prior for the deterministic
  Möbius/harmonic worlds; bootstrap SMC with resampling for hidden drift. With
  enough particles this is the best predictor any observation-only learner can be;
* the open-loop predictor: the prior pushed through the KNOWN actions, never
  looking at outcomes. Whatever it gains is knowledge of the dynamics, not memory;
* the receipt's own 48-particle physics filter, recomputed exactly.

NumPy only. Classical simulation throughout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .mobius import (
    DRIFT_STD, PHASES, PORTS, harmonic_step, matched_particle_filter, pulse,
    trajectories, wrap,
)
from .sequence import log_loss

WORLDS = ('mobius', 'harmonic', 'drift')


def prior_shapes(rng, n):
    """n independent draws from qword.mobius.start_shape's distribution, vectorized."""
    mean = rng.uniform(-np.pi, np.pi, n)
    concentration = rng.uniform(.12, 2.2, n)
    return wrap(mean[:, None] + rng.vonmises(0., concentration[:, None], (n, PHASES)))


def port_probabilities(theta, action):
    """qword.mobius.observe_probability for an (n, PHASES) array of constellations."""
    phi = PORTS[int(action)]
    z = np.exp(1j*np.asarray(theta))
    m1 = z.mean(-1)*np.exp(-1j*phi)
    m2 = (z*z).mean(-1)*np.exp(-2j*phi)
    return np.clip(.5 + .37*m1.real + .09*m2.real, .04, .96)


def advance(theta, action, world, rng):
    phi = float(PORTS[int(action)])
    if world == 'mobius':
        return pulse(theta, phi)
    if world == 'harmonic':
        return harmonic_step(theta, phi)
    if world == 'drift':
        return wrap(pulse(theta, phi) + rng.normal(0, DRIFT_STD, size=np.shape(theta)))
    raise ValueError('world must be mobius|harmonic|drift')


def _normalized(log_weight):
    w = np.exp(log_weight - log_weight.max())
    return w/w.sum()


def filter_predictions(actions, outcomes, world, particles=5000, seed=0, initial=None):
    """Bayes-filter and open-loop predictions of P(y_t=1 | actions, earlier outcomes).

    `initial`, if given, is an (n_sequences, n_particles, PHASES) array of starting
    particles that replaces draws from the prior (used by tests). Returns a dict
    with 'bayes' and 'open_loop' arrays shaped like `actions`, plus the median
    effective sample size of the Bayes weights at the end of each sequence.
    """
    actions = np.asarray(actions)
    outcomes = np.asarray(outcomes)
    if actions.shape != outcomes.shape or actions.ndim != 2:
        raise ValueError('actions and outcomes must be matching 2-D arrays')
    if world not in WORLDS:
        raise ValueError('world must be mobius|harmonic|drift')
    if initial is None and particles < 1:
        raise ValueError('particles must be positive')
    rng = np.random.default_rng(seed)
    free_rng = np.random.default_rng([seed, 1])  # open-loop noise never sees resampling draws
    bayes = np.empty(actions.shape)
    open_loop = np.empty(actions.shape)
    final_ess = []
    for i in range(len(actions)):
        start = prior_shapes(rng, particles) if initial is None else np.asarray(initial[i], dtype=float)
        n = len(start)
        theta = start.copy()
        log_weight = np.zeros(n)
        # Hidden drift is stochastic, so the open-loop cloud needs its own propagation.
        free = start.copy() if world == 'drift' else None
        for t, a in enumerate(actions[i]):
            p = port_probabilities(theta, a)
            bayes[i, t] = _normalized(log_weight) @ p
            open_loop[i, t] = (p if free is None else port_probabilities(free, a)).mean()
            if world == 'drift':
                log_weight += np.log(p if outcomes[i, t] else 1 - p)
                w = _normalized(log_weight)
                if 1/np.sum(w**2) < n/2:
                    theta = theta[rng.choice(n, size=n, p=w)]
                    log_weight[:] = 0.
                theta = advance(theta, a, world, rng)
                free = advance(free, a, world, free_rng)
            else:
                # Deterministic dynamics: no resampling needed, and the open loop is
                # simply uniform weights on the same particles.
                log_weight += np.log(p if outcomes[i, t] else 1 - p)
                theta = advance(theta, a, world, rng)
        w = _normalized(log_weight)
        final_ess.append(float(1/np.sum(w**2)))
    return {'bayes': bayes, 'open_loop': open_loop, 'median_final_ess': float(np.median(final_ess))}


def decompose(worlds=WORLDS, seeds=(2026, 2027, 2028), test=80, length=24, particles=5000,
              receipt_particles=48, convergence_particles=(1000, 20000)):
    """Score oracle / Bayes ceiling / open loop / receipt filter on v3's test splits."""
    runs = []
    for world in worlds:
        for seed in seeds:
            start = perf_counter()
            a, y, oracle = trajectories(world, test, length, seed*13 + 3)
            main = filter_predictions(a, y, world, particles, seed*31 + 7)
            receipt_filter = matched_particle_filter(a, y, world, seed=seed*17 + 9, particles=receipt_particles)
            row = {'world': world, 'seed': seed,
                   'oracle_hidden_state': log_loss(oracle, y),
                   'uniform': log_loss(np.full(y.shape, .5), y),
                   f'bayes_ceiling_{particles}': log_loss(main['bayes'], y),
                   f'open_loop_{particles}': log_loss(main['open_loop'], y),
                   f'receipt_physics_filter_{receipt_particles}': log_loss(receipt_filter, y),
                   f'bayes_median_final_ess_{particles}': main['median_final_ess']}
            if world == 'mobius':
                for n in convergence_particles:
                    row[f'bayes_ceiling_{n}'] = log_loss(filter_predictions(a, y, world, n, seed*31 + 8)['bayes'], y)
            row['seconds'] = round(perf_counter() - start, 1)
            runs.append(row)
            print(world, seed, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    summary = {}
    for world in worlds:
        rows = [r for r in runs if r['world'] == world]
        keys = [k for k in rows[0] if k not in ('world', 'seed', 'seconds')]
        summary[world] = {k: {'mean': float(np.mean([r[k] for r in rows])),
                              'per_seed': [float(r[k]) for r in rows]} for k in keys}
        ceiling = summary[world][f'bayes_ceiling_{particles}']['mean']
        uniform = summary[world]['uniform']['mean']
        oracle = summary[world]['oracle_hidden_state']['mean']
        summary[world]['share_of_oracle_lead_not_in_observations'] = (ceiling - oracle)/(uniform - oracle)
        summary[world]['share_of_ceiling_reached_open_loop'] = (
            (uniform - summary[world][f'open_loop_{particles}']['mean'])/(uniform - ceiling))
    return {'protocol': {'worlds': list(worlds), 'seeds': list(seeds), 'test_sequences': test, 'length': length,
                         'test_split': 'qword.mobius.trajectories(world, test, length, seed*13+3), as results/mobius-v3.json',
                         'particles': particles, 'receipt_particles': receipt_particles,
                         'convergence_particles_mobius': list(convergence_particles),
                         'bayes': 'true law + disclosed prior; importance sampling (mobius, harmonic) or bootstrap SMC (drift)',
                         'open_loop': 'same prior and law, pushed through the known actions, outcomes never used',
                         'claim': 'classical simulation; the oracle sees hidden phases and is not a learner target'},
            'runs': runs, 'summary': summary}


def main():
    parser = argparse.ArgumentParser(description='Q-word v3 ceiling decomposition')
    parser.add_argument('--worlds', nargs='+', choices=WORLDS, default=list(WORLDS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[2026, 2027, 2028])
    parser.add_argument('--test', type=int, default=80)
    parser.add_argument('--length', type=int, default=24)
    parser.add_argument('--particles', type=int, default=5000)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    receipt = decompose(args.worlds, args.seeds, args.test, args.length, args.particles)
    text = json.dumps(receipt, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n', encoding='utf8')
    for world, s in receipt['summary'].items():
        print(world, {k: round(v['mean'], 5) if isinstance(v, dict) else round(v, 3) for k, v in s.items()})


if __name__ == '__main__':
    main()
