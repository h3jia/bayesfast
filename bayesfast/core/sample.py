from .density import *
from distributed import Pub, Sub, Client, get_client, LocalCluster
from ..utils import random as bfrandom
from ..samplers import NUTS, THMC, NTrace, TTrace, HTrace, ETrace, TraceTuple
from ..utils import threadpool_limits, check_client, all_isinstance
import numpy as np
import warnings
from inspect import isclass

__all__ = ['sample']

# TODO: use tqdm to rewrite sampling progress report
# TODO: add the option of multiprocessing.Pool back
# TODO: add saving results every x iterations
# TODO: fix multi-threading
# TODO: fix pub/sub key


def sample(density, prior=None, trace=None, sampler='NUTS', n_run=None, client=None,
           verbose=True, n_update=None):
    # DEVELOPMENT NOTES
    # if use_surrogate is not specified in density_options
    # x_0 is interpreted as in original space and will be transformed
    # otherwise, x_0 is understood as in the specified space
    if not isinstance(density, (Density, DensityLite)):
        raise ValueError('density should be a Density or DensityLite.')
    if isinstance(trace, NTrace):
        sampler = 'NUTS'
    if isinstance(trace, TTrace):
        sampler = 'THMC'
    elif isinstance(trace, (HTrace, ETrace)):
        raise NotImplementedError
    elif trace is None:
        if sampler == 'NUTS':
            trace = NTrace()
        elif sampler == 'THMC':
            trace = TTrace()
        elif sampler == 'HMC' or sampler == 'Ensemble':
            raise NotImplementedError
        else:
            raise ValueError('unexpected value for sampler.')
    elif isinstance(trace, TraceTuple):
        sampler = trace.sampler
        if sampler == 'NUTS' or sampler == 'THMC':
            pass
        elif sampler == 'HMC' or sampler == 'Ensemble':
            raise NotImplementedError
        else:
            raise ValueError('unexpected value for trace.sampler.')
    else:
        raise ValueError('unexpected value for trace.')
    
    if isinstance(trace, (NTrace, TTrace)) and trace.x_0 is None:
        dim = density.input_size
        trace._x_0 = bfrandom.multivariate_normal(np.zeros(dim), np.eye(dim),
                                                  trace.n_chain)
    
    try:
        client, new_client = check_client(client)
        # dask_key = bfrandom.string()
        dask_key = 'BayesFast-' + client.id
        sub = Sub(dask_key)
        finished = 0
        
        if sampler == 'NUTS':
            def nested_helper(trace, i):
                """Without this, there will be an UnboundLocalError."""
                if isinstance(trace, NTrace):
                    trace._set_chain_id(i)
                elif isinstance(trace, TraceTuple):
                    trace = trace.traces[i]
                else:
                    raise RuntimeError('unexpected type for trace.')
                return trace
            def nuts_worker(i):
                try:
                    with threadpool_limits(1):
                        _trace = nested_helper(trace, i)
                        def logp_and_grad(x):
                            return density.logp_and_grad(
                                x, original_space=not _trace.transform_x)
                        nuts = NUTS(logp_and_grad=logp_and_grad, trace=_trace, 
                                    dask_key=dask_key)
                        t = nuts.run(n_run, verbose)
                        if t.transform_x:
                            t._samples_original = density.to_original(t.samples)
                            t._logp_original = density.to_original_density(
                                t.logp, x_trans=t.samples)
                    return t
                except:
                    pub = Pub(dask_key)
                    pub.put(['Error', i])
                    raise
            
            foo = client.map(nuts_worker, range(trace.n_chain))
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
                if finished == trace.n_chain:
                    break
            tt = client.gather(foo)
            return TraceTuple(tt)
        
        elif sampler == 'THMC':
            if not isinstance(prior, (Density, DensityLite)):
                raise ValueError('prior should be a Density or DensityLite.')
            def nested_helper(trace, i):
                """Without this, there will be an UnboundLocalError."""
                if isinstance(trace, TTrace):
                    trace._set_chain_id(i)
                elif isinstance(trace, TraceTuple):
                    trace = trace.traces[i]
                else:
                    raise RuntimeError('unexpected type for trace.')
                return trace
            def nuts_worker(i):
                try:
                    with threadpool_limits(1):
                        _trace = nested_helper(trace, i)
                        def logp_and_grad(x):
                            return density.logp_and_grad(x, original_space=not _trace.transform_x)
                        def logprior_and_grad(x):
                            return prior.logp_and_grad(x, original_space=not _trace.transform_x)
                        nuts = THMC(logp_and_grad=logp_and_grad, logprior_and_grad=logprior_and_grad, trace=_trace, dask_key=dask_key)
                        t = nuts.run(n_run, verbose, n_update)
                        if t.transform_x:
                            t._samples_original = density.to_original(t.samples)
                            t._logp_original = density.to_original_density(
                                t.logp, x_trans=t.samples)
                    return t
                except:
                    pub = Pub(dask_key)
                    pub.put(['Error', i])
                    raise
            
            foo = client.map(nuts_worker, range(trace.n_chain))
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
                if finished == trace.n_chain:
                    break
            tt = client.gather(foo)
            return TraceTuple(tt)
        
        elif sampler == 'HMC' or sampler == 'Ensemble':
            raise NotImplementedError
        
        else:
            raise RuntimeError('unexpected value for sampler.')
    finally:
        if new_client:
            client.cluster.close()
            client.close()
