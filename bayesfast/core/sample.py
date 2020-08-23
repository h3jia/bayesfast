from .density import *
from ..utils.sobol import multivariate_normal
from ..utils.parallel import ParallelBackend, get_backend
from ..utils.random import get_generator
from ..samplers import NUTS, HMC, TNUTS, THMC
from ..samplers import NTrace, HTrace, TNTrace, THTrace, ETrace
from ..samplers import SampleTrace, TraceTuple
from threadpoolctl import threadpool_limits
import numpy as np
import warnings
from inspect import isclass
from multiprocess import Manager
try:
    from distributed import Pub, Sub
    HAS_DASK = True
except Exception:
    HAS_DASK = False

__all__ = ['sample']

# TODO: use tqdm to rewrite sampling progress report
# TODO: add saving results every x iterations
# TODO: fix multi-threading


def sample(density, sample_trace=None, sampler='NUTS', n_run=None,
           parallel_backend=None, verbose=True):
    if not isinstance(density, (Density, DensityLite)):
        raise ValueError('density should be a Density or DensityLite.')
    
    if isinstance(sample_trace, NTrace):
        sampler = 'NUTS'
    elif isinstance(sample_trace, HTrace):
        sampler = 'HMC'
    elif isinstance(sample_trace, TNTrace):
        sampler = 'TNUTS'
    elif isinstance(sample_trace, THTrace):
        sampler = 'THMC'
    elif isinstance(sample_trace, ETrace):
        raise NotImplementedError
    elif sample_trace is None or isinstance(sample_trace, dict):
        sample_trace = {} if (sample_trace is None) else sample_trace
        if sampler == 'NUTS':
            sample_trace = NTrace(**sample_trace)
        elif sampler == 'HMC':
            sample_trace = HTrace(**sample_trace)
        elif sampler == 'TNUTS':
            sample_trace = TNTrace(**sample_trace)
        elif sampler == 'THMC':
            sample_trace = THTrace(**sample_trace)
        elif sampler == 'Ensemble':
            raise NotImplementedError
        else:
            raise ValueError('unexpected value for sampler.')
    elif isinstance(sample_trace, TraceTuple):
        sampler = sample_trace.sampler
        if any(sampler == _ for _ in ('NUTS', 'HMC', 'TNUTS', 'THMC')):
            pass
        elif sampler == 'Ensemble':
            raise NotImplementedError
        else:
            raise ValueError('unexpected value for sample_trace.sampler.')
    else:
        raise ValueError('unexpected value for sample_trace.')
    
    if isinstance(sample_trace, SampleTrace):
        if sample_trace.random_generator is None:
            sample_trace.random_generator = get_generator()
            get_generator().normal()
        if sample_trace.x_0 is None:
            dim = density.input_size
            if dim is None:
                raise RuntimeError('Neither SampleTrace.x_0 nor Density'
                                   '/DensityLite.input_size is defined.')
            sample_trace._x_0 = multivariate_normal(
                np.zeros(dim), np.eye(dim), sample_trace.n_chain)
            sample_trace._x_0_transformed = True
        elif not sample_trace.x_0_transformed:
            sample_trace._x_0 = density.from_original(sample_trace._x_0)
            sample_trace._x_0_transformed = True
    
    if parallel_backend is None:
        parallel_backend = get_backend()
    else:
        parallel_backend = ParallelBackend(parallel_backend)
    
    if parallel_backend.kind == 'multiprocess':
        use_dask = False
        dask_key = None
        process_lock = Manager().Lock()
    elif parallel_backend.kind == 'dask':
        if not HAS_DASK:
            raise RuntimeError('you want me to use dask but have not installed '
                               'it.')
        use_dask = True
        dask_key = 'BayesFast-' + parallel_backend.backend.id
        process_lock = None
        sub = Sub(dask_key)
        finished = 0
    else:
        raise RuntimeError('unexpected value for parallel_backend.kind.')
    
    with parallel_backend:
        if any(sampler == _ for _ in ('NUTS', 'HMC', 'TNUTS', 'THMC')):
            def nested_helper(sample_trace, i):
                """Without this, there will be an UnboundLocalError."""
                if isinstance(sample_trace, SampleTrace):
                    sample_trace._init_chain(i)
                elif isinstance(sample_trace, TraceTuple):
                    sample_trace = sample_trace.sample_traces[i]
                else:
                    raise RuntimeError('unexpected type for sample_trace.')
                return sample_trace
            
            def _sampler_worker(i, sampler_class):
                try:
                    with threadpool_limits(1):
                        _sample_trace = nested_helper(sample_trace, i)
                        def logp_and_grad(x):
                            return density.logp_and_grad(
                                x, original_space=False)
                        _sampler = sampler_class(
                            logp_and_grad=logp_and_grad,
                            sample_trace=_sample_trace, dask_key=dask_key,
                            process_lock=process_lock)
                        t = _sampler.run(n_run, verbose)
                        t._samples_original = density.to_original(t.samples)
                        t._logp_original = density.to_original_density(
                            t.logp, x_trans=t.samples)
                    return t
                except Exception:
                    if use_dask:
                        pub = Pub(dask_key)
                        pub.put(['Error', i])
                    raise
            
            if sampler == 'NUTS':
                sampler_worker = lambda i: _sampler_worker(i, NUTS)
            elif sampler == "HMC":
                sampler_worker = lambda i: _sampler_worker(i, HMC)
            elif sampler == 'TNUTS':
                sampler_worker = lambda i: _sampler_worker(i, TNUTS)
            elif sampler == 'THMC':
                sampler_worker = lambda i: _sampler_worker(i, THMC)
            else:
                raise RuntimeError('unexpected value for sampler.')
            
            if use_dask:
                foo = parallel_backend.map_async(sampler_worker,
                                                 range(sample_trace.n_chain))
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
                tt = parallel_backend.gather(foo)
            else:
                tt = parallel_backend.map(sampler_worker,
                                          range(sample_trace.n_chain))
            return TraceTuple(tt)
        
        elif sampler == 'Ensemble':
            raise NotImplementedError
        
        else:
            raise RuntimeError('unexpected value for sampler.')
