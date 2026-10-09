"""Trained-to-convergence checks of the v2 and v3 rankings.

Same data generators, models, learning rates and seeding style as the original
runners, with three changes that the original receipts' own falsifiers ask for:

1. long training (default 3000 updates) with the best-validation weights kept;
2. a larger data regime for v3 (3,200 training / 400 validation sequences);
3. the extended model set, which adds `gru_action_readout` beside the original `gru`.

Test splits are exactly those of results/mobius-v3.json (v3) and
results/sequence-v2-seed*.json (v2). Classical CPU simulation throughout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from . import sequence as seq
from .mobius import trajectories

REGIMES = {
    'v3': {'published': {'train': (160, 1), 'valid': (35, 2)},
           'large': {'train': (3200, 101), 'valid': (400, 102)}},
    'v2': {'published': {'train': (350, 1), 'valid': (60, 2)}},
}
SPLITS = {'v3': {'items': ('mobius', 'harmonic', 'drift'), 'test': 80, 'length': 24, 'batch': 40},
          'v2': {'items': ('qubit', 'hmm'), 'test': 110, 'length': 20, 'batch': 48}}


def data(suite, item, regime, seed):
    """(train, valid, test) splits. v3 seeds follow qword.mobius.run_benchmark; v2 seeds follow qword.sequence.experiment."""
    spec, split = REGIMES[suite][regime], SPLITS[suite]
    if suite == 'v3':
        make = lambda n, offset: trajectories(item, n, split['length'], seed*13 + offset)
        return make(*spec['train']), make(*spec['valid']), make(split['test'], 3)
    make = lambda n, offset: seq.sequences(item, n, split['length'], seed + offset)
    return make(*spec['train']), make(*spec['valid']), make(split['test'], 3)


def run(suite='v3', items=None, regimes=('published',), seeds=(2026, 2027, 2028), steps=3000, every=100,
        names=None):
    if seq.torch is None:
        raise RuntimeError("pip install -e '.[sequence]' (requires PyTorch)")
    torch = seq.torch
    torch.set_num_threads(1)
    catalogue = seq.extended_models()
    names = tuple(names or catalogue)
    items = tuple(items or SPLITS[suite]['items'])
    runs = []
    for regime in regimes:
        for item in items:
            for seed in seeds:
                train, valid, test = data(suite, item, regime, seed)
                row = {'suite': suite, 'item': item, 'regime': regime, 'seed': seed,
                       'train_sequences': len(train[0]), 'valid_sequences': len(valid[0]),
                       'baselines': {'oracle': seq.log_loss(test[2], test[1]),
                                     'uniform': seq.log_loss(np.full(test[1].shape, .5), test[1]),
                                     'one_step_tabulation': seq.log_loss(seq.markov_baseline(train, test), test[1])},
                       'models': {}}
                for name in names:
                    offset = sum(map(ord, name))
                    torch.manual_seed(seed + offset)
                    model = catalogue[name]()
                    start = perf_counter()
                    history, best_step = seq.fit_best(model, train, valid, steps, SPLITS[suite]['batch'],
                                                      seed + 101 + offset,
                                                      lr=.012 if name == 'quantum_instrument' else .006, every=every)
                    probs = seq.predicted(model, test[0], test[1])
                    row['models'][name] = {
                        'test_nll': seq.log_loss(probs, test[1]),
                        'mae_from_oracle': float(np.mean(np.abs(probs - test[2]))),
                        'best_step': best_step,
                        'trainable_parameters': int(sum(p.numel() for p in model.parameters())),
                        'valid_curve': [[h['step'], round(h['valid_nll'], 6)] for h in history],
                        'train_seconds_local': round(perf_counter() - start, 1)}
                    print(suite, regime, item, seed, name, round(row['models'][name]['test_nll'], 5),
                          'best step', best_step, flush=True)
                runs.append(row)
    return {'schema': 'qword.converged.v1',
            'protocol': {'suite': suite, 'regimes': {r: REGIMES[suite][r] for r in regimes}, 'seeds': list(seeds),
                         'steps': steps, 'validate_every': every, 'models': list(names),
                         'selection': 'weights with the lowest validation NLL; test split untouched until the end',
                         'learning_rates': 'quantum_instrument 0.012, others 0.006 (as published)',
                         'seeding': 'torch.manual_seed(seed+sum(ord(name))), minibatch rng seed+101+sum(ord(name))',
                         'split': SPLITS[suite],
                         'claim': 'classical CPU simulation; equal updates are not equal parameters, FLOPs or memory'},
            'runs': runs}


def summarize(receipt):
    """Mean and per-seed test NLL for every suite/regime/item/model."""
    table = {}
    for row in receipt['runs']:
        key = f"{row['suite']}/{row['regime']}/{row['item']}"
        cell = table.setdefault(key, {'seeds': [], 'baselines': {}, 'models': {}})
        cell['seeds'].append(row['seed'])
        for name, value in row['baselines'].items():
            cell['baselines'].setdefault(name, []).append(value)
        for name, result in row['models'].items():
            entry = cell['models'].setdefault(name, {'test_nll': [], 'best_step': []})
            entry['test_nll'].append(result['test_nll'])
            entry['best_step'].append(result['best_step'])
    for cell in table.values():
        cell['baselines'] = {k: {'mean': float(np.mean(v)), 'per_seed': v} for k, v in cell['baselines'].items()}
        cell['models'] = {k: {'mean_test_nll': float(np.mean(v['test_nll'])), 'per_seed_test_nll': v['test_nll'],
                              'best_step': v['best_step']} for k, v in cell['models'].items()}
    return table


def merge(paths):
    parts = [json.loads(Path(p).read_text()) for p in paths]
    merged = {'schema': 'qword.converged.v1', 'protocol': [p['protocol'] for p in parts],
              'runs': [r for p in parts for r in p['runs']]}
    merged['summary'] = summarize(merged)
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('v3', 'v2'), default='v3')
    parser.add_argument('--items', nargs='+', help='v3 worlds or v2 tasks (default: all)')
    parser.add_argument('--regimes', nargs='+', default=['published'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[2026, 2027, 2028])
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--every', type=int, default=100)
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--merge', nargs='+', type=Path, help='combine partial receipts instead of training')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.merge:
        receipt = merge(args.merge)
    else:
        bad = [r for r in args.regimes if r not in REGIMES[args.suite]]
        if bad:
            parser.error(f'unknown regime(s) for {args.suite}: {bad}')
        receipt = run(args.suite, args.items, args.regimes, args.seeds, args.steps, args.every, args.models)
        receipt['summary'] = summarize(receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=1, sort_keys=True) + '\n', encoding='utf8')
    for key, cell in receipt['summary'].items():
        print(key, {k: round(v['mean_test_nll'], 5) for k, v in cell['models'].items()})


if __name__ == '__main__':
    main()
