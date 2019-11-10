from .density import *
from distributed import Sub, Client, get_client, LocalCluster
from ..utils.random import *
from ..samplers import NUTS
from ..samplers.hmc_utils import Trace
import numpy as np
import warnings


# TODO: use tqdm to rewrite sampling progress report
# TODO: enable multi-threading

def sample(density, client=None, n_chain=4, n_iter=None, n_warmup=None,
           trace=None, random_state=None, x_0=None, verbose=True,
           full_return=False, density_options={}, sampler_options={},
           sampler='NUTS'):
    if not isinstance(density, (Density, DensityLite)):
        raise ValueError('density should be a Density or DensityLite.')
    if isinstance(trace, Trace):
        trace = [trace]
    if (hasattr(trace, '__iter__') and len(trace) > 0 and 
        all(isinstance(t, Trace) for t in trace)):
        n_chain = len(trace)
    else:
        try:
            n_chain = int(n_chain)
            assert n_chain > 0
        except:
            raise ValueError('invalid value for n_chain')
        trace = [None for i in range(n_chain)]
    
    if n_iter is None:
        n_iter = 3000 if (trace[0] is None) else 1000
    else:
        try:
            n_iter = int(n_iter)
            assert n_iter > 0
        except:
            raise ValueError('invalid value for n_iter.')
    if n_warmup is None:
        n_warmup = 1000 if (trace[0] is None) else 0
    else:
        try:
            n_warmup = int(n_warmup)
            assert n_warmup > 0
        except:
            raise ValueError('invalid value for n_warmup.')
    
    if trace[0] is None:
        if hasattr(random_state, '__iter__'):
            random_state = [check_state(rs) for rs in random_state]
            if len(random_state) < n_chain:
                raise ValueError('you did not give me enough random_state(s).')
        else:
            random_state = check_state(random_state)
            random_state = split_state(random_state, n_chain)
        if x_0 is None:
            dim = density.input_size
            x_0 = random_multivariate_normal(
                np.zeros(dim), np.eye(dim), n_chain)
        else:
            x_0 = np.atleast_2d(x_0)
            if x_0.shape[0] < n_chain:
                raise ValueError('you did not give me enough x_0(s).')
            x_0 = x_0[:n_chain, :]
    else:
        random_state = [None for i in range(n_chain)]
        x_0 = [None for i in range(n_chain)]
    
    try:
        density_options = dict(density_options).copy()
    except:
        raise ValueError('density_options should be a dict.')
    if not 'use_surrogate' in density_options:
        density_options['use_surrogate'] = True
    if not 'original_space' in density_options:
        density_options['original_space'] = False
    
    try:
        _new_client = False
        if client is None:
            try:
                client = get_client()
            except:
                cluster = LocalCluster(threads_per_worker=1)
                client = Client(cluster)
                _new_client = True
        else:
            if not isinstance(client, Client):
                raise ValueError('invalid value for client.')
        dask_key = random_str()
        sub = Sub(dask_key)
        finished = 0
        
        if sampler == 'NUTS':
            def nuts_worker(i):
                def logp_and_grad(x):
                    return density.logp_and_grad(x, **density_options)
                nuts = NUTS(
                    logp_and_grad=logp_and_grad, trace=trace[i], 
                    dask_key=dask_key, chain_id=i, random_state=random_state[i],
                    x_0=x_0[i], **sampler_options)
                t = nuts.run(n_iter, n_warmup, verbose)
                return t if full_return else t.get()
            foo = client.map(nuts_worker, range(n_chain), pure=False)
            for msg in sub:
                if not hasattr(msg, '__iter__'):
                    warnings.warn('unexpected message: {}.'.format(msg),
                                  RuntimeWarning)
                elif isinstance(msg[0], Warning):
                    warnings.warn(msg[1], msg[0])
                elif msg[0] == 'SamplingProceeding':
                    print(msg[1])
                elif msg[0] == 'SamplingFinished':
                    print(msg[1])
                    finished += 1
                else:
                    warnings.warn('unexpected message: {}.'.format(msg),
                                  RuntimeWarning)
                if finished == n_chain:
                    break
            return np.asarray(client.gather(foo))
                
        elif sampler == 'HMC':
            raise NotImplementedError
        
        else:
            raise ValueError(
                'Sorry I do not know how to do {}.'.format(sampler))
    finally:
        if _new_client:
            client.close()
            cluster.close()
