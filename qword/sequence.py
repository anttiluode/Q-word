"""Q-word v2: controlled predictive-state sequence benchmark, CPU-only.

Quantum models are simulated with real Bloch vectors. Their state evolution
satisfies qubit positivity constraints, but NO quantum hardware is used.

The baseline named real-operator is a learned NONLINEAR bounded real-state
recurrence. It is not claimed to be an exact linear PSR / OOM.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # optional heavyweight sequence benchmark
    torch = None
    nn = None
    F = None

ACTIONS = 3


def rotate_numpy(r, vector):
    theta = np.linalg.norm(vector)
    if theta < 1e-12:
        return r.copy()
    n = vector / theta
    return r * np.cos(theta) + np.cross(n, r) * np.sin(theta) + n * (n @ r) * (1 - np.cos(theta))


def quantum_step(r, a, y):
    """One weak instrument outcome and conditional unitary, entirely in Bloch form."""
    axes = np.array([[1., 0., 0.], [0., 1., 0.], [.4, .3, np.sqrt(.75)]])
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    eta = np.array([.84, .78, .72])[a]
    n = axes[a]
    s = 1 - 2 * int(y)
    dot = n @ r
    p = (1 + eta * dot) / 2
    updated = (np.sqrt(1 - eta**2) * (r - dot*n) + (dot + s*eta)*n) / (1 + s*eta*dot)
    vec = np.array([.24 * (a+1), .31 * (1-2*y), .20 * (a-1)])
    return p if y == 0 else (1-p), rotate_numpy(updated, vec)


def classical_step(b, a, y):
    """Three-state controlled hidden Markov filter and conditional observation law."""
    prob1 = np.array([[.88, .18, .62], [.12, .82, .35], [.66, .25, .87]])[a]
    emit = prob1 if y else 1 - prob1
    p = float(b @ emit)
    posterior = b * emit / p
    eye = np.eye(3)
    cycle = np.roll(eye, 1, axis=1)
    reverse = np.roll(eye, -1, axis=1)
    transition = [.96 * eye + .04/3, .58*eye + .42*cycle, .58*eye + .42*reverse][a]
    return p, posterior @ transition


def sequences(task: str, n: int, length: int, seed: int):
    """Independent trajectories; return actions, observations, oracle P(Y=1)."""
    if task not in ('qubit', 'hmm') or n <= 0 or length <= 0:
        raise ValueError('task=qubit|hmm and positive size required')
    rng = np.random.default_rng(seed)
    actions = rng.integers(ACTIONS, size=(n, length), dtype=np.int64)
    outcome = np.empty((n, length), dtype=np.int64)
    oracle = np.empty((n, length), dtype=np.float32)
    for i in range(n):
        r = np.array([.1, -.2, .4]) if task == 'qubit' else np.array([.35, .2, .45])
        for t, a in enumerate(actions[i]):
            if task == 'qubit':
                p0, _ = quantum_step(r, int(a), 0)
                p1 = 1 - p0
            else:
                p1, _ = classical_step(r, int(a), 1)
            oracle[i, t] = p1
            y = int(rng.random() < p1)
            outcome[i, t] = y
            _, r = quantum_step(r, int(a), y) if task == 'qubit' else classical_step(r, int(a), y)
    return actions, outcome, oracle


def log_loss(probs: np.ndarray, targets: np.ndarray) -> float:
    p = np.clip(np.asarray(probs, dtype=float), 1e-8, 1-1e-8)
    y = np.asarray(targets, dtype=float)
    return float(np.mean(-y*np.log(p) - (1-y)*np.log1p(-p)))


if torch is not None:
    def batch_rotate(r, vector):
        # torch Rodrigues formula; stable at vector=0 and works with gradients.
        angle = torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(1e-7)
        n = vector / angle
        cos = angle.cos()
        sin = angle.sin()
        return cos*r + sin*torch.cross(n, r, dim=-1) + (1-cos)*n*(n*r).sum(-1, keepdim=True)


    class QuantumInstrument(nn.Module):
        """Trainable physically valid single-qubit instrument with conditional rotation."""
        def __init__(self):
            super().__init__()
            self.axes = nn.Parameter(torch.randn(ACTIONS,3))
            self.sharpness = nn.Parameter(torch.zeros(ACTIONS))
            self.rotation = nn.Parameter(torch.randn(ACTIONS,2,3)*.17)
            self.initial = nn.Parameter(torch.zeros(3))

        def forward(self, actions, observations):
            b, length = actions.shape
            init_length = torch.linalg.vector_norm(self.initial).clamp_min(1e-8)
            r = (self.initial * (torch.tanh(init_length) / init_length))[None].expand(b,-1)
            n = F.normalize(self.axes, dim=-1)
            eta = .98*torch.sigmoid(self.sharpness)
            predictions=[]
            for t in range(length):
                a, y = actions[:,t], observations[:,t]
                direction, strength = n[a], eta[a,None]
                projection=(r*direction).sum(-1,keepdim=True)
                p0 = (.5*(1+strength*projection)).clamp(1e-6,1-1e-6)
                predictions.append(1-p0.squeeze(-1))
                sign = (1 - 2*y).to(r.dtype)[:,None]
                nr = (torch.sqrt(1-strength**2)*(r-projection*direction)
                      +(projection+sign*strength)*direction)/(1+sign*strength*projection)
                r = batch_rotate(nr,self.rotation[a,y])
                # Numerical roundoff can move r very slightly outside unit ball.
                r=r / torch.linalg.vector_norm(r,dim=-1,keepdim=True).clamp_min(1.)
            return torch.stack(predictions,dim=1)


    class RealOperator(nn.Module):
        """Bounded classical operator recurrence with same 3-component state size.

        Uses unconstrained matrices followed by tanh. NOT a linear PSR or OOM.
        """
        def __init__(self, width=3):
            super().__init__()
            self.matrix = nn.Parameter(torch.eye(width)[None,None].repeat(ACTIONS,2,1,1) + torch.randn(ACTIONS,2,width,width)*.1)
            self.shift = nn.Parameter(torch.zeros(ACTIONS,2,width))
            self.read = nn.Parameter(torch.randn(ACTIONS,width)*.15)
            self.bias = nn.Parameter(torch.zeros(ACTIONS))
            self.initial = nn.Parameter(torch.zeros(width))

        def forward(self, actions, observations):
            b,length=actions.shape
            r=self.initial[None].expand(b,-1)
            out=[]
            for t in range(length):
                a,y=actions[:,t],observations[:,t]
                out.append(torch.sigmoid((self.read[a]*r).sum(-1)+self.bias[a]))
                r=torch.tanh(torch.bmm(self.matrix[a,y],r.unsqueeze(-1)).squeeze(-1)+self.shift[a,y])
            return torch.stack(out,1)


    class SmallGRU(nn.Module):
        """Ordinary nonquantum recurrent predictor; consumes the same data."""
        def __init__(self, width=4):
            super().__init__()
            self.cell=nn.GRUCell(ACTIONS+2,width)
            self.read=nn.Linear(width+ACTIONS,1)
            self.initial=nn.Parameter(torch.zeros(width))

        def forward(self,actions,observations):
            b,length=actions.shape
            r=self.initial[None].expand(b,-1)
            out=[]
            for t in range(length):
                a=F.one_hot(actions[:,t],ACTIONS).float()
                out.append(torch.sigmoid(self.read(torch.cat([r,a],-1)).squeeze(-1)))
                o=F.one_hot(observations[:,t],2).float()
                r=self.cell(torch.cat([a,o],-1),r)
            return torch.stack(out,1)


    class WindowTransformer(nn.Module):
        """One-layer causal transformer with explicit window (no earlier leakage)."""
        def __init__(self, width=8, window=8):
            super().__init__()
            self.window=window
            self.current_action=nn.Embedding(ACTIONS,width)
            self.previous_action=nn.Embedding(ACTIONS+1,width)
            self.previous_outcome=nn.Embedding(3,width)
            self.position=nn.Embedding(128,width)
            self.layer=nn.TransformerEncoderLayer(d_model=width,nhead=2,dim_feedforward=2*width,batch_first=True,dropout=0.,activation='gelu',norm_first=False)
            self.out=nn.Linear(width,1)

        def forward(self, actions,observations):
            b,length=actions.shape
            prev_a=torch.cat([torch.full_like(actions[:,:1],ACTIONS),actions[:,:-1]],1)
            prev_o=torch.cat([torch.full_like(observations[:,:1],2),observations[:,:-1]],1)
            seq=self.current_action(actions)+self.previous_action(prev_a)+self.previous_outcome(prev_o)
            seq=seq+self.position(torch.arange(length,device=actions.device))[None]
            ids=torch.arange(length,device=actions.device)
            mask=(ids[None,:]>ids[:,None]) | (ids[:,None]-ids[None,:]>=self.window)
            z=self.layer(seq,src_mask=mask)
            return torch.sigmoid(self.out(z).squeeze(-1))


    def models():
        return {'quantum_instrument': QuantumInstrument, 'real_operator': RealOperator,
                'gru': SmallGRU, 'window_transformer': WindowTransformer}


    def predicted(model, actions, outcomes, batch=128):
        model.eval()
        results=[]
        with torch.no_grad():
            for i in range(0,len(actions),batch):
                a=torch.as_tensor(actions[i:i+batch],dtype=torch.long)
                y=torch.as_tensor(outcomes[i:i+batch],dtype=torch.long)
                results.append(model(a,y).numpy())
        return np.concatenate(results)


    def fit(model, train, valid, steps, batch_size, seed, lr=.006):
        rng=np.random.default_rng(seed)
        opt=torch.optim.Adam(model.parameters(),lr=lr)
        model.train()
        history=[]
        for k in range(steps):
            indices=rng.integers(0,len(train[0]),size=batch_size)
            a=torch.as_tensor(train[0][indices],dtype=torch.long)
            y=torch.as_tensor(train[1][indices],dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            p=model(a,y).clamp(1e-6,1-1e-6)
            loss=F.binary_cross_entropy(p,y.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),2.)
            opt.step()
            if (k+1)%max(1,steps//4)==0:
                history.append({'step':k+1,'train_nll':float(loss.detach()),'valid_nll':log_loss(predicted(model,*valid[:2]),valid[1])})
                model.train()
        return history


def markov_baseline(train, test):
    """Current action, previous action, previous outcome, with Laplace smoothing."""
    a,y,_=train
    counts=np.ones((ACTIONS,ACTIONS+1,3,2),dtype=float)
    previous=np.concatenate([np.full((len(a),1),2),y[:,:-1]],axis=1)
    prev_a=np.concatenate([np.full((len(a),1),ACTIONS),a[:,:-1]],axis=1)
    np.add.at(counts,(a.flatten(),prev_a.flatten(),previous.flatten(),y.flatten()),1)
    counts/=counts.sum(-1,keepdims=True)
    a,y,_=test
    previous=np.concatenate([np.full((len(a),1),2),y[:,:-1]],axis=1)
    prev_a=np.concatenate([np.full((len(a),1),ACTIONS),a[:,:-1]],axis=1)
    return counts[a,prev_a,previous,1]


def experiment(task='qubit', seed=2026, train_size=650, valid_size=100, test_size=160,
               length=24, steps=180, batch_size=48):
    if torch is None:
        raise ImportError('PyTorch required. Install: pip install -e .[sequence]')
    torch.set_num_threads(1)
    train=sequences(task,train_size,length,seed+1)
    valid=sequences(task,valid_size,length,seed+2)
    test=sequences(task,test_size,length,seed+3)
    baselines={'uniform':log_loss(np.full_like(test[2],.5),test[1]),
               'one_step_markov':log_loss(markov_baseline(train,test),test[1]),
               'oracle':log_loss(test[2],test[1])}
    result={'task':task,'seed':seed,'data':{'train_sequences':train_size,'valid_sequences':valid_size,
          'test_sequences':test_size,'sequence_length':length,'actions':3,'outcomes':2},
          'train_steps_per_model':steps,'minibatch':batch_size,'nll_nats_per_token':baselines,'models':{}}
    for j,(name,constructor) in enumerate(models().items()):
        torch.manual_seed(seed+100*j)
        model=constructor()
        num_params=sum(p.numel() for p in model.parameters())
        start=time.perf_counter()
        history=fit(model,train,valid,steps,batch_size,seed+200+j,lr=.012 if name=='quantum_instrument' else .006)
        elapsed=time.perf_counter()-start
        probs=predicted(model,test[0],test[1])
        # Evaluate all predictions AND latter half for longer-history sensitivity.
        scores={'nll':log_loss(probs,test[1]),
                'late_nll':log_loss(probs[:,length//2:],test[1][:,length//2:]),
                'mae_from_oracle':float(np.mean(abs(probs-test[2]))),
                'prob_range':[float(probs.min()),float(probs.max())]}
        result['models'][name]={'trainable_parameters':num_params,'live_state_real_scalars':3 if name in ('quantum_instrument','real_operator') else (4 if name=='gru' else 8*min(length,8)),
           'scores':scores,'validation_history':history,'train_wall_seconds_local':round(elapsed,3)}
    result['interpretation']='Synthetic model comparison only; quantum_instrument is computed classically in three real Bloch coordinates. No quantum computing advantage implied. Parameter counts and inference costs are not equal.'
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=['qubit','hmm','both'],default='both')
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--steps',type=int,default=180)
    parser.add_argument('--train',type=int,default=650)
    parser.add_argument('--valid',type=int,default=100)
    parser.add_argument('--test',type=int,default=160)
    parser.add_argument('--length',type=int,default=24)
    parser.add_argument('--output',default=None)
    args=parser.parse_args(argv)
    tasks=['qubit','hmm'] if args.task=='both' else [args.task]
    report={'schema':'qword.sequence.v2','configuration':vars(args),
            'experiments':{task:experiment(task,args.seed,args.train,args.valid,args.test,args.length,args.steps) for task in tasks}}
    encoded=json.dumps(report,indent=2)
    if args.output:
        path=Path(args.output)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(encoded+'\n',encoding='utf8')
        print('Wrote',path)
    for task,r in report['experiments'].items():
        print('\n',task, 'oracle:',round(r['nll_nats_per_token']['oracle'],4),'one-step:',round(r['nll_nats_per_token']['one_step_markov'],4))
        for name,m in r['models'].items():
            print(f"  {name:21s} NLL {m['scores']['nll']:.4f} MAE oracle {m['scores']['mae_from_oracle']:.4f} params {m['trainable_parameters']}")
    return report


if __name__=='__main__':
    main()
