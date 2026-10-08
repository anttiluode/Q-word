"""Independent Möbius oscillator transfer worlds for Q-word.

All worlds are CLASSICAL simulated oscillators. Models see action indices and past
binary observations only; the simulator's phases are evaluation-only. The exact
Möbius odometer knows the initial phases and applied actions and is a privileged
known-law reference, not a fair learned baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

PHASES = 16
PORTS = np.array([0., 2*np.pi/3, 4*np.pi/3])
ACTION_STRENGTH = 0.11
SECOND_HARMONIC = 0.28
DRIFT_STD = 0.075


def wrap(theta):
    return (np.asarray(theta) + np.pi) % (2*np.pi) - np.pi


def pulse(theta, phi, duration=ACTION_STRENGTH):
    """Exact dtheta/dt=sin(phi-theta) flow. A single SU(1,1) Möbius map."""
    d = wrap(np.asarray(theta) - phi)
    return wrap(phi + 2*np.arctan2(np.exp(-duration)*np.sin(d/2), np.cos(d/2)))


def group_step(matrix, phi, duration=ACTION_STRENGTH):
    """Compose a disk automorphism with observer odometer (2x2 complex)."""
    c = np.tanh(duration / 2.) * np.exp(1j*phi)
    M = np.array([[1., c], [np.conj(c), 1.]], dtype=np.complex128)
    M /= np.sqrt(1-np.abs(c)**2)
    return M @ matrix


def apply_group(theta, matrix):
    z = np.exp(1j*np.asarray(theta))
    z2 = (matrix[0,0]*z + matrix[0,1])/(matrix[1,0]*z + matrix[1,1])
    return np.angle(z2)


def crossratios(theta):
    """Real projective invariants; complex expression checked via sine ratios.

    Distinct angles are required; use an arctan statistic against ill-conditioning.
    """
    t = np.asarray(theta)
    a, b, c = t[:3]
    d = t[3:]
    return (np.sin((a-c)/2)*np.sin((b-d)/2) /
            (np.sin((a-d)/2)*np.sin((b-c)/2)))


def harmonic_step(theta, phi, duration=ACTION_STRENGTH, epsilon=SECOND_HARMONIC, steps=12):
    """Deterministic first+second harmonic ODE with RK4 (not a Möbius map)."""
    theta = np.asarray(theta, dtype=float).copy()
    h = duration/steps
    def f(x):
        delta = phi-x
        return np.sin(delta)+epsilon*np.sin(2*delta)
    for _ in range(steps):
        k1=f(theta); k2=f(theta+h*k1/2)
        k3=f(theta+h*k2/2); k4=f(theta+h*k3)
        theta += h*(k1+2*k2+2*k3+k4)/6
    return wrap(theta)


def observe_probability(theta, action):
    """Noisy binary observation with fixed, disclosed 1st/2nd-harmonic reader."""
    z = np.exp(1j*np.asarray(theta))
    phi = PORTS[int(action)]
    m1 = np.mean(z)*np.exp(-1j*phi)
    m2 = np.mean(z*z)*np.exp(-2j*phi)
    return float(np.clip(.5+.37*m1.real+.09*m2.real, .04, .96))


def start_shape(rng):
    """Independent, unseen constellations with an instrument-known distribution.

    No model gets the particular initial phases. Nondegenerate shapes are ensured
    for projective-invariant diagnostics, not to tailor task to a model.
    """
    mean = rng.uniform(-np.pi,np.pi)
    concentration = rng.uniform(.12,2.2)
    return wrap(mean + rng.vonmises(0,concentration,PHASES))


def step(theta, action, world, rng=None):
    phi=float(PORTS[int(action)])
    if world == 'mobius':
        return pulse(theta,phi)
    if world == 'harmonic':
        return harmonic_step(theta,phi)
    if world == 'drift':
        if rng is None:
            raise ValueError('drift requires random generator')
        # Hidden, heterogeneous drift AFTER each known exact read. The observer
        # is not told these offsets; it is not an SU(1,1) controlled action.
        return wrap(pulse(theta,phi) + rng.normal(0,DRIFT_STD,size=len(theta)))
    raise ValueError('world must be mobius|harmonic|drift')


def trajectories(world, n, length, seed, return_shape=False):
    """No hidden phase observations leak into action/observation training data.

    For all worlds the p_t is measured BEFORE action t changes the state.
    Outcomes are sampled before the transition and do NOT influence it.
    Initial constellations in train/valid/test are independent seeded draws.
    """
    if n<=0 or length<=0 or world not in ('mobius','harmonic','drift'):
        raise ValueError('bad world or nonpositive trajectory dimensions')
    rng = np.random.default_rng(seed)
    actions = rng.integers(0,3,size=(n,length),dtype=np.int64)
    outcomes=np.empty((n,length),dtype=np.int64)
    oracle=np.empty((n,length),dtype=np.float32)
    shapes=[]
    for i in range(n):
        phases=start_shape(rng)
        shapes.append(phases.copy())
        for t, a in enumerate(actions[i]):
            p=observe_probability(phases,int(a))
            oracle[i,t]=p
            outcomes[i,t]=int(rng.random()<p)
            phases=step(phases,int(a),world,rng)
    if return_shape:
        return actions,outcomes,oracle,np.stack(shapes)
    return actions,outcomes,oracle


def matched_particle_filter(actions, observations, world, seed=900, particles=128):
    """Non-neural, action/observation-only Bayesian filter with privileged law/prior.

    Initial shape is unknown: independent particles drawn from the SAME disclosed
    initial-state distribution. No teacher oracle phases are passed into here.
    A resampling particle filter tracks histories; noise in world C is sampled
    independently as a valid bootstrap approximation to hidden drift.
    """
    if particles < 16:
        raise ValueError('particles must be >=16')
    if world not in ('mobius','harmonic','drift'):
        raise ValueError('unknown world')
    rng=np.random.default_rng(seed)
    actions=np.asarray(actions)
    observations=np.asarray(observations)
    if actions.shape != observations.shape or actions.ndim != 2:
        raise ValueError('actions and observations must be matching 2-D arrays')
    result=np.empty(actions.shape,float)
    for i in range(len(actions)):
        prior=np.stack([start_shape(rng) for _ in range(particles)])
        weight=np.ones(particles)/particles
        for t,a in enumerate(actions[i]):
            phi=PORTS[int(a)]
            z=np.exp(1j*prior)
            m1=np.mean(z,axis=-1)*np.exp(-1j*phi)
            m2=np.mean(z*z,axis=-1)*np.exp(-2j*phi)
            pp=np.clip(.5+.37*m1.real+.09*m2.real,.04,.96)
            result[i,t]=float(weight@pp)
            # Update *after* predicting the current observation.
            like=pp if observations[i,t] else 1-pp
            weight=weight*like
            weight/=weight.sum()
            if 1/np.sum(weight**2) < particles/2:
                indices=rng.choice(particles,size=particles,p=weight)
                prior=prior[indices]
                weight[:]=1/particles
            if world == 'mobius':
                prior=pulse(prior,phi)
            elif world == 'harmonic':
                prior=harmonic_step(prior,phi)
            else:
                prior=wrap(pulse(prior,phi)+rng.normal(0,DRIFT_STD,size=prior.shape))
    return result


def odometer_diagnostic(seed=4100, reads=32):
    rng=np.random.default_rng(seed)
    # Deterministic distinct constellation; no random near-coincidence singularities.
    phases=wrap(np.linspace(-np.pi,np.pi,PHASES,endpoint=False) + .025*np.sin(np.arange(PHASES)*1.8))
    actions=rng.integers(0,3,size=reads)
    matrix=np.eye(2,dtype=complex)
    advanced=phases.copy()
    for a in actions:
        advanced=pulse(advanced,PORTS[a])
        matrix=group_step(matrix,PORTS[a])
    reconstructed=apply_group(phases,matrix)
    inverse=np.linalg.inv(matrix)
    restored=apply_group(advanced,inverse)
    mobius_shape=float(np.max(np.abs(np.arctan(crossratios(advanced))-np.arctan(crossratios(phases)))))
    harmonic=phases.copy()
    for a in actions:
        harmonic=harmonic_step(harmonic,PORTS[a])
    harmonic_shape=float(np.median(np.abs(np.arctan(crossratios(harmonic))-np.arctan(crossratios(phases)))))
    drift=phases.copy()
    for a in actions:
        drift=step(drift,a,'drift',rng)
    blind_restored=apply_group(drift,inverse)
    return {'seed':seed, 'reads':reads,
            'group_reconstruction_max_rad':float(np.max(np.abs(wrap(reconstructed-advanced)))),
            'group_inverse_max_rad':float(np.max(np.abs(wrap(restored-phases)))),
            'mobius_shape_max_atan_crossratio':mobius_shape,
            'harmonic_shape_median_atan_crossratio':harmonic_shape,
            'hidden_drift_inverse_rms_rad':float(np.sqrt(np.mean(wrap(blind_restored-phases)**2)))}


def aliased_pair_diagnostic():
    """Equal first moments but distinct second moments; deterministic strong probe."""
    A=np.array([np.pi/3,-np.pi/3,np.pi/3,-np.pi/3])
    B=np.array([0.,0.,0.,np.pi])
    m=lambda x,k:complex(np.mean(np.exp(1j*k*x)))
    return {'m1_gap':float(abs(m(A,1)-m(B,1))),
            'm2_gap':float(abs(m(A,2)-m(B,2))),
            'post_ping_m1_gap':float(abs(m(pulse(A,0.,.25),1)-m(pulse(B,0.,.25),1)))}


def run_benchmark(worlds=('mobius','harmonic','drift'),seeds=(2026,2027,2028),
                  train=200,valid=45,test=100,length=24,steps=90,batch=40,models_to_run=None,particles=128):
    """Train the existing four Q-word v2 models. Equal observation rights, NOT
    equal parameter counts or compute. For screening only, not discovery claims.
    """
    from .sequence import torch, models, fit, predicted, markov_baseline, log_loss
    if torch is None:
        raise RuntimeError("pip install -e '.[sequence]' (requires PyTorch)")
    torch.set_num_threads(1)
    names=tuple(models_to_run or models())
    if any(name not in models() for name in names):
        raise ValueError('unknown model')
    records=[]
    for world in worlds:
        for seed in seeds:
            train_data=trajectories(world,train,length,seed*13+1)
            valid_data=trajectories(world,valid,length,seed*13+2)
            test_data=trajectories(world,test,length,seed*13+3)
            # Oracle sees full simulator phases: privileged upper reference.
            reference={'oracle_hidden_state':log_loss(test_data[2],test_data[1]),
                       'uniform':log_loss(np.full_like(test_data[2],.5),test_data[1]),
                       'one_step_tabulation':log_loss(markov_baseline(train_data,test_data),test_data[1])}
            start_filter=perf_counter()
            filter_pred=matched_particle_filter(test_data[0],test_data[1],world,
                                                seed=seed*17+9,particles=particles)
            reference['physics_particle_filter']=log_loss(filter_pred,test_data[1])
            reference['physics_particle_filter_oracle_mae']=float(np.mean(abs(filter_pred-test_data[2])))
            reference['physics_particle_filter_seconds']=round(perf_counter()-start_filter,3)
            for name in names:
                torch.manual_seed(seed+sum(map(ord,name)))
                model=models()[name]()
                params=sum(p.numel() for p in model.parameters())
                start=perf_counter()
                history=fit(model,train_data,valid_data,steps,batch,seed+101+sum(map(ord,name)),
                            lr=.012 if name=='quantum_instrument' else .006)
                estimates=predicted(model,*test_data[:2])
                reference[name]=log_loss(estimates,test_data[1])
                reference[name+'_oracle_mae']=float(np.mean(abs(estimates-test_data[2])))
                reference[name+'_seconds']=round(perf_counter()-start,3)
                reference[name+'_params']=params
            records.append({'world':world,'seed':seed,'metrics':reference})
    summary={}
    for world in worlds:
        matches=[r['metrics'] for r in records if r['world']==world]
        measures=[k for k in matches[0] if not k.endswith(('_params','_seconds'))]
        summary[world]={k:{'mean':float(np.mean([m[k] for m in matches])),
                           'std':float(np.std([m[k] for m in matches],ddof=1)) if len(matches)>1 else None}
                        for k in measures}
    return {'protocol':{'seeds':list(seeds),'worlds':list(worlds),'train':train,'valid':valid,
                        'test':test,'length':length,'steps':steps,'batch':batch,
                        'models':list(names),'particles':particles,'action_strength':ACTION_STRENGTH,
                        'second_harmonic':SECOND_HARMONIC,'drift_std':DRIFT_STD,
                        'readout':'0.5+0.37 Re(e^-i phi m1)+0.09 Re(e^-2i phi m2)',
                        'access':'learners and particle filter get actions and past Bernoulli observations only; oracle sees phases',
                        'claim':'exploratory synthetic; equal updates not equal FLOPs/params; no quantum speedup'},
            'diagnostics':{'odometer':[odometer_diagnostic(s) for s in seeds],
                           'alias':aliased_pair_diagnostic()},
            'runs':records,'summary':summary}


def main():
    parser=argparse.ArgumentParser(description='Q-word Mobius independent transfer benchmark')
    parser.add_argument('--worlds',nargs='+',choices=('mobius','harmonic','drift'),default=['mobius','harmonic','drift'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[2026,2027,2028])
    parser.add_argument('--train',type=int,default=200)
    parser.add_argument('--valid',type=int,default=45)
    parser.add_argument('--test',type=int,default=100)
    parser.add_argument('--length',type=int,default=24)
    parser.add_argument('--steps',type=int,default=90)
    parser.add_argument('--batch',type=int,default=40)
    parser.add_argument('--models',nargs='+',choices=('quantum_instrument','real_operator','gru','window_transformer'))
    parser.add_argument('--particles',type=int,default=128)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    receipt=run_benchmark(args.worlds,args.seeds,args.train,args.valid,args.test,args.length,args.steps,args.batch,args.models,args.particles)
    s=json.dumps(receipt,indent=2,sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(s+'\n',encoding='utf8')
    for world,models in receipt['summary'].items():
        print(world, {k:round(v['mean'],5) for k,v in models.items() if not k.endswith('_oracle_mae')})
    print('diagnostics',receipt['diagnostics'])

if __name__=='__main__':
    main()
