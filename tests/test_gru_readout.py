"""Why the original GRU cannot read per-port state, and checks of the fixed readout."""
import unittest

import numpy as np

from qword.sequence import torch


@unittest.skipIf(torch is None, 'PyTorch optional dependency missing')
class ReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def _logit_gaps(self, model, state, read):
        """logit(action 0) - logit(action 1) for a given hidden state."""
        logits = []
        for action in (0, 1):
            onehot = torch.nn.functional.one_hot(torch.tensor([action]), 3).float()
            logits.append(read(model, state, action, onehot))
        return float(logits[0] - logits[1])

    def test_original_gru_logit_gap_between_actions_ignores_the_state(self):
        from qword.sequence import SmallGRU
        torch.manual_seed(0)
        model = SmallGRU()
        read = lambda m, r, a, onehot: m.read(torch.cat([r, onehot], -1)).squeeze()
        gaps = [self._logit_gaps(model, torch.randn(1, 4), read) for _ in range(20)]
        self.assertLess(np.ptp(gaps), 1e-6)

    def test_action_readout_gru_logit_gap_depends_on_the_state(self):
        from qword.sequence import ActionReadoutGRU
        torch.manual_seed(0)
        model = ActionReadoutGRU()
        read = lambda m, r, a, onehot: (m.read[a]*r[0]).sum() + m.bias[a]
        gaps = [self._logit_gaps(model, torch.randn(1, 4), read) for _ in range(20)]
        self.assertGreater(np.ptp(gaps), 1e-2)

    def test_registries(self):
        from qword.sequence import models, extended_models
        self.assertEqual(list(models()), ['quantum_instrument', 'real_operator', 'gru', 'window_transformer'])
        self.assertEqual(list(extended_models()), list(models()) + ['gru_action_readout'])

    def test_action_readout_gru_is_causal_and_trainable(self):
        from qword.sequence import ActionReadoutGRU
        torch.manual_seed(3)
        model = ActionReadoutGRU()
        a = torch.tensor([[0, 1, 2, 0, 2, 1, 0, 1, 2]])
        y = torch.tensor([[0, 0, 1, 1, 0, 0, 1, 1, 1]])
        changed = y.clone()
        changed[:, 5:] = 1 - changed[:, 5:]
        torch.testing.assert_close(model(a, y)[:, :6], model(a, changed)[:, :6])
        probs = model(a, y)
        torch.nn.functional.binary_cross_entropy(probs, y.float()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_fit_best_restores_the_best_validation_weights(self):
        from qword.sequence import ActionReadoutGRU, fit_best, predicted, log_loss
        from qword.mobius import trajectories
        train, valid = trajectories('mobius', 40, 8, 5), trajectories('mobius', 20, 8, 6)
        torch.manual_seed(1)
        model = ActionReadoutGRU()
        history, best_step = fit_best(model, train, valid, steps=40, batch_size=8, seed=2, every=10)
        best = min(h['valid_nll'] for h in history)
        self.assertEqual(best_step, min(h['step'] for h in history if h['valid_nll'] == best))
        self.assertAlmostEqual(log_loss(predicted(model, *valid[:2]), valid[1]), best, places=6)

    def test_converged_runner_smoke(self):
        from qword.converged import run, summarize
        receipt = run('v3', items=('mobius',), regimes=('published',), seeds=(2026,), steps=4, every=2,
                      names=('real_operator', 'gru_action_readout'))
        summary = summarize(receipt)
        cell = summary['v3/published/mobius']
        self.assertEqual(set(cell['models']), {'real_operator', 'gru_action_readout'})
        self.assertEqual(receipt['runs'][0]['train_sequences'], 160)
        self.assertTrue(all(np.isfinite(v['mean_test_nll']) for v in cell['models'].values()))


if __name__ == '__main__':
    unittest.main()
