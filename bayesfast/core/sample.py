from .density import *
from distributed import Sub, Client, get_client, LocalCluster
from ..utils import random as bfrandom
from ..samplers import NUTS
from ..samplers.hmc_utils import Trace
from ..utils import threadpool_limits, check_client, all_isinstance
import numpy as np
import warnings
from inspect import isclass

__all__ = ['sample']

# TODO: use tqdm to rewrite sampling progress report
# TODO: fix multi-threading
# TODO: add wrapper for emcee
# TODO: fix overwriting options
# TODO: fix pub/sub key
# TODO: use previous samples to determine initial mass


def sample(density, client=None, n_chain=4, n_iter=None, n_warmup=None,
           trace=None, random_state=None, x_0=None, verbose=True,
           return_trace=False, sampler_options={}, sampler='NUTS'):
    # DEVELOPMENT NOTES
    # if use_surrogate is not specified in density_options
    # x_0 is interpreted as in original space and will be transformed
    # otherwise, x_0 is understood as in the specified space
    if not isinstance(density, (Density, DensityLite)):
        raise ValueError('density should be a Density or DensityLite.')
    if isinstance(trace, Trace):
        trace = [trace]
    if (hasattr(trace, '__iter__') and len(trace) > 0 and
        all_isinstance(trace, Trace)):
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
    
    try:
        density_options = dict(density_options).copy()
    except:
        raise ValueError('density_options should be a dict.')
    if isinstance(density, Density) and not 'use_surrogate' in density_options:
        density_options['use_surrogate'] = True
    if not 'original_space' in density_options:
        density_options['original_space'] = False
        _transform_x = True
    else:
        _transform_x = False
    
    if trace[0] is None:
        if hasattr(random_state, '__iter__'):
            random_state = [bfrandom.check_state(rs) for rs in random_state]
            if len(random_state) < n_chain:
                raise ValueError('you did not give me enough random_state(s).')
        else:
            random_state = bfrandom.check_state(random_state)
            random_state = bfrandom.split_state(random_state, n_chain)
        if x_0 is None:
            dim = density.input_size
            x_0 = bfrandom.multivariate_normal(
                np.zeros(dim), np.eye(dim), n_chain)
        else:
            x_0 = np.atleast_2d(x_0)
            if x_0.shape[0] < n_chain:
                raise ValueError('you did not give me enough x_0(s).')
            x_0 = x_0[:n_chain, :]
            if _transform_x:
                x_0 = density.from_original(x_0)
    else:
        random_state = [None for i in range(n_chain)]
        x_0 = [None for i in range(n_chain)]
    
    try:
        client, _new_client = check_client(client)
        # dask_key = bfrandom.string()
        dask_key = 'BayesFast-' + client.id
        sub = Sub(dask_key)
        finished = 0
        
        if sampler == 'NUTS':
            def nuts_worker(i):
                with threadpool_limits(1):
                    def logp_and_grad(x):
                        return density.logp_and_grad(x, **density_options)
                    nuts = NUTS(logp_and_grad=logp_and_grad, trace=trace[i], 
                                dask_key=dask_key, chain_id=i, 
                                random_state=random_state[i], x_0=x_0[i], 
                                **sampler_options)
                    t = nuts.run(n_iter, n_warmup, verbose)
                return t# if return_trace else t.get()
            foo = client.map(nuts_worker, range(n_chain))
            for msg in sub:
                if not hasattr(msg, '__iter__'):
                    warnings.warn('unexpected message: {}.'.format(msg),
                                  RuntimeWarning)
                elif msg[0] == 'Error':
                    break
                elif isclass(msg[0]) and issubclass(msg[0], Warning):
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
            tt = client.gather(foo)
            if _transform_x:
                xx = np.array([density.to_original(t.get()) for t in tt])
            else:
                xx = np.array([t.get() for t in tt])
            return (xx, tt) if return_trace else xx
                
        elif sampler == 'HMC':
            raise NotImplementedError
        
        elif sampler == 'EnsembleSampler':
            raise NotImplementedError
        
        else:
            raise ValueError(
                'Sorry I do not know how to do {}.'.format(sampler))
    finally:
        if _new_client:
            client.cluster.close()
            client.close()


class Sampler:
    
    def __init__(self, method='NUTS'):
        raise NotImplementedError