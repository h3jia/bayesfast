import numpy as np
from collections import namedtuple
from ..trace import _HTrace
from .integration import CpuLeapfrogIntegrator
from .stats import NStepStats
from ...utils import random as bfrandom
import warnings
from copy import deepcopy
import time
from distributed import Pub

__all__ = ['BaseHMC']

# TODO: review the code


HMCStepData = namedtuple("HMCStepData", 
                         "end, accept_stat, divergence_info, stats")


DivergenceInfo = namedtuple("DivergenceInfo", "message, exec_info, state")


class BaseHMC:
    """Superclass to implement Hamiltonian/hybrid monte carlo."""
    def __init__(self, logp_and_grad, trace, dask_key=None):
        self._logp_and_grad = logp_and_grad
        self.dask_key = dask_key
        if isinstance(trace, self._expected_trace):
            self._trace = trace
        else:
            raise ValueError('trace should be a HTrace or NTrace.')
        self._chain_id = trace.chain_id
        self._prefix = ' CHAIN #' + str(self._chain_id) + ' : '
        self.integrator = CpuLeapfrogIntegrator(self._trace.metric,
                                                logp_and_grad)
        try:
            logp_0, grad_0 = logp_and_grad(trace.x_0)
            assert np.isfinite(logp_0).all() and np.isfinite(grad_0).all()
        except:
            raise ValueError('failed to get finite logp and/or grad at x_0.')
    
    _expected_trace = _HTrace
    
    def _hamiltonian_step(self, start, p0, step_size):
        """Compute one hamiltonian trajectory and return the next state.

        Subclasses must overwrite this method and return a `HMCStepData`.
        """
        raise NotImplementedError("Abstract method")

    def astep(self):
        """Perform a single HMC iteration."""
        try:
            q0 = self._trace._samples[-1]
        except:
            q0 = self._trace.x_0
            assert q0.ndim == 1
        p0 = self._trace.metric.random(self._trace.random_state)
        start = self.integrator.compute_state(q0, p0)

        if not np.isfinite(start.energy):
            self._trace.metric.raise_ok()
            raise RuntimeError(
                "Bad initial energy, please check the Hamiltonian at p = {}, "
                "q = {}.".format(p0, q0))
            
        step_size = self._trace.step_size.current(self.warmup)
        hmc_step = self._hamiltonian_step(start, p0, step_size)
        self._trace.step_size.update(hmc_step.accept_stat, self.warmup)
        self._trace.metric.update(hmc_step.end.q, self.warmup)
        step_stats = NStepStats(**hmc_step.stats, 
                                **self._trace.step_size.sizes(), 
                                warmup=self.warmup, 
                                diverging=bool(hmc_step.divergence_info))
        self._trace.update(hmc_step.end.q, step_stats)
    
    def run(self, n_run=None, verbose=True, n_update=None):
        if self._dask_key is not None:
            pub = Pub(self._dask_key)
            def sw(message, category, *args, **kwargs):
                pub.put([category, self._prefix + str(message)])
            warnings.showwarning = sw
        try:            
            i_iter = self.trace.i_iter
            n_iter = self.trace.n_iter
            n_warmup = self.trace.n_warmup
            if n_run is None:
                n_run = n_iter - i_iter
            else:
                try:
                    n_run = int(n_run)
                    assert n_run > 0
                except:
                    raise ValueError(self._prefix + 'invalid value for n_run.')
                if n_run > n_iter - i_iter:
                    warnings.warn(
                        self._prefix + 'n_run is larger than n_iter-i_iter. '
                        'Set n_run=n_iter-i_iter for now.', RuntimeWarning)
                    n_run = n_iter - i_iter
            if verbose:
                if n_update is None:
                    n_update = n_run // 5
                else:
                    try:
                        n_update = int(n_update)
                        assert n_update > 0
                    except:
                        warnings.warn(
                            self._prefix + 'invalid value for n_update. Using '
                            'n_run//5 for now.', RuntimeWarning)
                        n_update = n_run // 5
                t_s = time.time()
                t_i = time.time()
            for i in range(i_iter, i_iter + n_run):
                if verbose:
                    if i > i_iter and not i % n_update:
                        t_d = time.time() - t_i
                        t_i = time.time()
                        n_div = np.sum(self.trace.stats._diverging[-n_update:])
                        msg_0 = (
                            self._prefix +  'sampling proceeding [ {} / {} ], '
                            'last {} samples used {:.2f} seconds'.format(i, 
                            n_iter, n_update, t_d))
                        if n_div / n_update > 0.05:
                            msg_1 = (', while divergence encountered in {} '
                                     'sample(s).'.format(n_div))
                        else:
                            msg_1 = '.'
                        if self.warmup:
                            msg_2 = ' (warmup)'
                        else:
                            msg_2 = ''
                        if self._dask_key is None:
                            print(msg_0 + msg_1 + msg_2)
                        else:
                            pub.put(
                                ['SamplingProceeding', msg_0 + msg_1 + msg_2])
                self.warmup = bool(i < n_warmup)
                self.astep()
            if verbose:
                t_f = time.time()
                msg = (self._prefix + 'sampling finished [ {} / {} ], '
                       'obtained {} samples in {:.2f} seconds.'.format(n_iter,
                       n_iter, n_run, t_f - t_s))
                if self._dask_key is None:
                    print(msg)
                else:
                    pub.put(['SamplingFinished', msg])
            return self.trace
        finally:
            warnings.showwarning = warnings._showwarning_orig
    
    @property
    def trace(self):
        return self._trace
    
    @property
    def dask_key(self):
        return self._dask_key
    
    @dask_key.setter
    def dask_key(self, key):
        if key is None:
            pass
        else:
            try:
                key = str(key)
            except:
                warnings.warn('invalid value for dask_key.', RuntimeWarning)
                key = None
        self._dask_key = key
    
    @property
    def chain_id(self):
        return self._chain_id
