from .density import *
from distributed import Pub, Sub, Client, get_client, LocalCluster
from ..utils import random as bfrandom
from ..samplers import NUTS, SampleTrace, NTrace, HTrace, ETrace, TraceTuple
from ..utils import threadpool_limits, check_client
import numpy as np
import warnings
from inspect import isclass

__all__ = ['sample']

# TODO: use tqdm to rewrite sampling progress report
# TODO: add the option of multiprocessing.Pool back
# TODO: add saving results every x iterations
# TODO: fix multi-threading
# TODO: fix pub/sub key


def sample(density, sample_trace=None, sampler='NUTS', n_run=None, client=None,
           verbose=True):
    if not isinstance(density, (Density, DensityLite)):
        raise ValueError('density should be a Density or DensityLite.')
    if isinstance(sample_trace, NTrace):
        sampler = 'NUTS'
    elif isinstance(sample_trace, (HTrace, ETrace)):
        raise NotImplementedError
    elif sample_trace is None or isinstance(sample_trace, dict):
        sample_trace = {} if (sample_trace is None) else sample_trace
        if sampler == 'NUTS':
            sample_trace = NTrace(**sample_trace)
        elif sampler == 'HMC' or sampler == 'Ensemble':
            raise NotImplementedError
        else:
            raise ValueError('unexpected value for sampler.')
    elif isinstance(sample_trace, TraceTuple):
        sampler = sample_trace.sampler
        if sampler == 'NUTS':
            pass
        elif sampler == 'HMC' or sampler == 'Ensemble':
            raise NotImplementedError
        else:
            raise ValueError('unexpected value for sample_trace.sampler.')
    else:
        raise ValueError('unexpected value for sample_trace.')
    
    if isinstance(sample_trace, SampleTrace):
        if sample_trace.x_0 is None:
            dim = density.input_size
            if dim is None:
                raise RuntimeError('Neither SampleTrace.x_0 nor Density'
                                   '/DensityLite.input_size is defined.')
            sample_trace._x_0 = bfrandom.multivariate_normal(
                np.zeros(dim), np.eye(dim), sample_trace.n_chain)
            sample_trace._x_0_transformed = True
        elif not sample_trace.x_0_transformed:
            sample_trace._x_0 = density.from_original(sample_trace._x_0)
            sample_trace._x_0_transformed = True
    
    try:
        client, new_client = check_client(client)
        # dask_key = bfrandom.string()
        dask_key = 'BayesFast-' + client.id
        sub = Sub(dask_key)
        finished = 0
        
        if sampler == 'NUTS':
            def nested_helper(sample_trace, i):
                """Without this, there will be an UnboundLocalError."""
                if isinstance(sample_trace, NTrace):
                    sample_trace._init_chain(i)
                elif isinstance(sample_trace, TraceTuple):
                    sample_trace = sample_trace.sample_traces[i]
                else:
                    raise RuntimeError('unexpected type for sample_trace.')
                return sample_trace
            def nuts_worker(i):
                try:
                    with threadpool_limits(1):
                        _sample_trace = nested_helper(sample_trace, i)
                        def logp_and_grad(x):
                            return density.logp_and_grad(x,
                                                         original_space=False)
                        nuts = NUTS(logp_and_grad=logp_and_grad,
                                    sample_trace=_sample_trace, 
                                    dask_key=dask_key)
                        t = nuts.run(n_run, verbose)
                        t._samples_original = density.to_original(t.samples)
                        t._logp_original = density.to_original_density(
                            t.logp, x_trans=t.samples)
                    return t
                except:
                    pub = Pub(dask_key)
                    pub.put(['Error', i])
                    raise
            
            foo = client.map(nuts_worker, range(sample_trace.n_chain))
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
                if finished == sample_trace.n_chain:
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
