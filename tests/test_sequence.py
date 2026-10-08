"""Data-generation and physics/casuality regression checks for the optional v2 benchmark."""
import unittest
import numpy as np
from qword.sequence import sequences, quantum_step, classical_step, log_loss, markov_baseline, torch


class DataTests(unittest.TestCase):
    def test_quantum_born_probabilities_and_norm(self):
        rng=np.random.default_rng(17)
        r=np.array([.1,-.2,.4])
        for _ in range(600):
            a=int(rng.integers(3))
            p0, r0=quantum_step(r,a,0)
            p1, r1=quantum_step(r,a,1)
            self.assertAlmostEqual(p0+p1,1.,places=12)
            self.assertTrue(0<p0<1)
            self.assertLessEqual(max(np.linalg.norm(r0),np.linalg.norm(r1)),1+1e-10)
            r=r0 if rng.random()<p0 else r1

    def test_classical_filter_preserves_simplex(self):
        b=np.array([.35,.2,.45])
        for t in range(60):
            p1,b=classical_step(b,t%3,t%2)
            self.assertTrue(0 < p1 < 1)
            self.assertAlmostEqual(b.sum(),1.,places=12)
            self.assertGreaterEqual(float(b.min()),0.)

    def test_disjoint_seeded_trajectories_and_oracle(self):
        for task in ('qubit','hmm'):
            a,y,p=sequences(task,12,10,123)
            a2,y2,p2=sequences(task,12,10,123)
            np.testing.assert_array_equal(a,a2)
            np.testing.assert_array_equal(y,y2)
            np.testing.assert_array_equal(p,p2)
            self.assertTrue(np.all((p>0)&(p<1)))
            self.assertTrue(np.isfinite(log_loss(p,y)))
            self.assertGreater(np.mean(abs(p-.5)),.02)
            mk=markov_baseline((a,y,p),(a,y,p))
            self.assertEqual(mk.shape,p.shape)
            self.assertTrue(np.all((mk>0)&(mk<1)))


@unittest.skipIf(torch is None,'PyTorch optional dependency missing')
class TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_four_forward_backward(self):
        from qword.sequence import models
        actions=torch.tensor([[0,1,2,0],[2,1,0,2]],dtype=torch.long)
        y=torch.tensor([[0,1,1,0],[1,0,1,1]],dtype=torch.long)
        for name,ctor in models().items():
            with self.subTest(model=name):
                model=ctor()
                probs=model(actions,y)
                self.assertEqual(tuple(probs.shape),(2,4))
                self.assertTrue(torch.all(torch.isfinite(probs)).item())
                self.assertTrue(torch.all((probs>0)&(probs<1)).item())
                torch.nn.functional.binary_cross_entropy(probs,y.float()).backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_no_future_target_leakage(self):
        from qword.sequence import models
        torch.manual_seed(12)
        a=torch.tensor([[0,1,2,0,2,1,0,1,2]],dtype=torch.long)
        y=torch.tensor([[0,0,1,1,0,0,1,1,1]],dtype=torch.long)
        altered_a=a.clone(); altered_y=y.clone()
        altered_a[:,6:]=(altered_a[:,6:]+1)%3
        altered_y[:,5:]=1-altered_y[:,5:]
        for name,ctor in models().items():
            with self.subTest(model=name):
                model=ctor()
                original=model(a,y)
                changed=model(altered_a,altered_y)
                torch.testing.assert_close(original[:,:5],changed[:,:5],atol=1e-6,rtol=1e-6)
                # Observe y_t only AFTER predicting y_t.
                flip=y.clone(); flip[:,3]=1-flip[:,3]
                torch.testing.assert_close(original[:,:4],model(a,flip)[:,:4],atol=1e-6,rtol=1e-6)

    def test_quantum_prediction_state_constraint(self):
        from qword.sequence import QuantumInstrument
        torch.manual_seed(42)
        model=QuantumInstrument()
        with torch.no_grad():
            model.initial.fill_(100.)
        # unconstrained initial vector would have norm sqrt(3)*100.
        r=model.initial
        norm=torch.linalg.vector_norm(r)
        physical=r*(torch.tanh(norm)/norm)
        self.assertLessEqual(float(torch.linalg.vector_norm(physical).detach()),1.0000001)
        a=torch.randint(0,3,(3,80)); y=torch.randint(0,2,(3,80))
        p=model(a,y)
        self.assertTrue(torch.all((p>0)&(p<1)).item())


if __name__=='__main__':
    unittest.main()

@unittest.skipIf(torch is None,'PyTorch optional dependency missing')
class PlantRecovery(unittest.TestCase):
    def test_planted_qubit_instrument_exactly_representable(self):
        """Teacher parameters reproduce the generator's hidden-state posterior."""
        from qword.sequence import QuantumInstrument
        a,y,oracle=sequences('qubit',5,14,101)
        m=QuantumInstrument()
        axes=np.array([[1,0,0],[0,1,0],[.4,.3,np.sqrt(.75)]],dtype=float)
        axes/=np.linalg.norm(axes,axis=1)[:,None]
        eta=np.array([.84,.78,.72])/.98
        truth=np.array([.1,-.2,.4]); radius=np.linalg.norm(truth)
        init=truth*np.arctanh(radius)/radius
        with torch.no_grad():
            m.axes.copy_(torch.as_tensor(axes,dtype=torch.float32))
            m.sharpness.copy_(torch.as_tensor(np.log(eta/(1-eta)),dtype=torch.float32))
            for ai in range(3):
                for yi in range(2):
                    m.rotation[ai,yi]=torch.tensor([.24*(ai+1),.31*(1-2*yi),.20*(ai-1)])
            m.initial.copy_(torch.tensor(init,dtype=torch.float32))
        with torch.no_grad():
            p=m(torch.tensor(a),torch.tensor(y)).numpy()
        np.testing.assert_allclose(p,oracle,atol=5e-7,rtol=5e-7)
