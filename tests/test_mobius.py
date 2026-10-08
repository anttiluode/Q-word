"""Independent exact-geometry, data-causality and transfer-model checks."""
import unittest
import numpy as np

from qword.mobius import (
    ACTION_STRENGTH, PORTS, wrap, pulse, group_step, apply_group, crossratios,
    harmonic_step, step, start_shape, observe_probability,
    trajectories, odometer_diagnostic, aliased_pair_diagnostic, matched_particle_filter,
)


class MobiusTests(unittest.TestCase):
    def test_exact_mobius_flow_and_inverse(self):
        phases=np.linspace(-2.8,2.8,16)
        for phi in PORTS:
            for a in (.01,.11,.6):
                advanced=pulse(phases,phi,a)
                np.testing.assert_allclose(wrap(pulse(advanced,phi,-a)-phases),0,atol=2e-14)
                M=group_step(np.eye(2,dtype=complex),phi,a)
                np.testing.assert_allclose(wrap(advanced-apply_group(phases,M)),0,atol=2e-14)

    def test_group_composition_crossratios_and_observer(self):
        r=odometer_diagnostic()
        self.assertLess(r['group_reconstruction_max_rad'],1e-11)
        self.assertLess(r['group_inverse_max_rad'],1e-11)
        self.assertLess(r['mobius_shape_max_atan_crossratio'],1e-10)
        self.assertGreater(r['harmonic_shape_median_atan_crossratio'],1e-7)
        self.assertGreater(r['hidden_drift_inverse_rms_rad'],1e-4)

    def test_harmonic_zero_reduces_exact_flow(self):
        theta=np.linspace(-2.6,2.7,16)
        np.testing.assert_allclose(wrap(harmonic_step(theta,.7,.11,epsilon=0)-pulse(theta,.7,.11)),0,atol=1e-9)

    def test_alias_current_moment_but_not_future(self):
        result=aliased_pair_diagnostic()
        self.assertLess(result['m1_gap'],1e-14)
        self.assertGreater(result['m2_gap'],1.4)
        self.assertGreater(result['post_ping_m1_gap'],.01)

    def test_reproducible_independently_seeded_and_causal_observation(self):
        for world in ('mobius','harmonic','drift'):
            data=trajectories(world,8,15,72,return_shape=True)
            repeated=trajectories(world,8,15,72,return_shape=True)
            for x,y in zip(data,repeated):
                np.testing.assert_array_equal(x,y)
            actions,obs,oracle,shape=data
            self.assertEqual(actions.shape,(8,15))
            self.assertEqual(shape.shape,(8,16))
            self.assertTrue(((oracle>.03)&(oracle<.97)).all())
            for i in range(8):
                self.assertAlmostEqual(float(oracle[i,0]),observe_probability(shape[i],actions[i,0]),places=6)
            unseen=trajectories(world,8,15,73,return_shape=True)
            self.assertGreater(np.max(np.abs(shape-unseen[3])),.1)

    def test_particle_filter_has_no_future_observation_leak(self):
        for world in ('mobius','harmonic','drift'):
            a,y,p=trajectories(world,2,7,41)
            first=matched_particle_filter(a,y,world,seed=3,particles=32)
            changed=y.copy(); changed[:,4:]=1-changed[:,4:]
            second=matched_particle_filter(a,changed,world,seed=3,particles=32)
            np.testing.assert_allclose(first[:,:5],second[:,:5],atol=0,rtol=0)
            self.assertTrue(((first>.03)&(first<.97)).all())

    def test_controlled_worlds_have_same_observation_port(self):
        rng=np.random.default_rng(50)
        theta=start_shape(rng)
        for world in ('mobius','harmonic','drift'):
            self.assertAlmostEqual(observe_probability(theta,1),observe_probability(theta,1))
            r=np.random.default_rng(50)
            updated=step(theta,1,world,r)
            self.assertEqual(len(updated),16)
            self.assertTrue(np.isfinite(updated).all())


class LearnedModelCausality(unittest.TestCase):
    def test_learning_interfaces_fit_existing_four_models(self):
        from qword.sequence import torch, models
        if torch is None:
            self.skipTest('requires optional torch')
        torch.set_num_threads(1)
        a,y,p=trajectories('mobius',3,8,11)
        for name,ctor in models().items():
            with self.subTest(name=name):
                torch.manual_seed(19)
                model=ctor()
                actions=torch.tensor(a)
                outcomes=torch.tensor(y)
                first=model(actions,outcomes)
                edited=outcomes.clone()
                edited[:,4:]=1-edited[:,4:]
                second=model(actions,edited)
                torch.testing.assert_close(first[:,:5],second[:,:5],atol=1e-6,rtol=1e-6)
                self.assertEqual(tuple(first.shape),(3,8))

if __name__=='__main__':
    unittest.main()
