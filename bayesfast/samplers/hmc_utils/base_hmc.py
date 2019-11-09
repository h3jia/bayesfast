import numpy as np
from collections import namedtuple
from .trace import Trace
from .integration import CpuLeapfrogIntegrator
from .stats import StepStats
from ...utils.random_utils import random_str
import warnings
from copy import deepcopy
import time
from distributed import Pub

__all__ = ['BaseHMC']


HMCStepData = namedtuple("HMCStepData", 
                         "end, accept_stat, divergence_info, stats")


DivergenceInfo = namedtuple("DivergenceInfo", "message, exec_info, state")


class BaseHMC:
    """Superclass to implement Hamiltonian/hybrid monte carlo."""

    def __init__(self, logp_and_grad, trace=None, dask_key=None, chain_id=None, 
                 random_state=None, x_0=None, step_size=0.25, 
                 adapt_step_size=True, metric=None, adapt_metric=True, 
                 Emax=1000, target_accept=0.8, gamma=0.05, k=0.75, t0=10):
        self._logp_and_grad = logp_and_grad
        self.dask_key = dask_key
        self.chain_id = chain_id
        if isinstance(trace, Trace):
            self._trace = trace
        else:
            x_0 = np.atleast_1d(x_0)
            if x_0.ndim != 1:
                raise ValueError('x_0 should be a 1-d array.')
            try:
                logp_0, _ = logp_and_grad(x_0)
                assert np.isfinite(logp_0)
            except:
                raise ValueError('failed to get finite logp at x0.')
            self._trace = Trace(x_0, logp_0, random_state, step_size, 
                                adapt_step_size, metric, adapt_metric, Emax, 
                                target_accept, gamma, k, t0)
        self.integrator = CpuLeapfrogIntegrator(self._trace.metric, 
                                                logp_and_grad)

    def _hamiltonian_step(self, start, p0, step_size):
        """Compute one hamiltonian trajectory and return the next state.

        Subclasses must overwrite this method and return a `HMCStepData`.
        """
        raise NotImplementedError("Abstract method")

    def astep(self):
        """Perform a single HMC iteration."""
        q0 = self._trace.samples[-1]
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
        step_stats = StepStats(**hmc_step.stats, 
                               **self._trace.step_size.sizes(), 
                               warmup=self.warmup, 
                               diverging=bool(hmc_step.divergence_info))
        self._trace.update(hmc_step.end.q, step_stats)
    
    def run(self, n_iter=3000, n_warmup=1000, verbose=True, n_update=None,
            return_copy=True):
        n_iter = int(n_iter)
        n_warmup = int(n_warmup)
        try:
            if self._dask_key is not None:
                pub = Pub(self._dask_key)
                def sw(message, category, *args, **kwargs):
                    pub.put([category, message])
                warnings.showwarning = sw
            if not n_iter >= 0:
                raise ValueError(self._prefix + 'n_iter cannot be negative.')
            if n_warmup > n_iter:
                warnings.warn(
                    self._prefix + 'n_warmup is larger than n_iter. Setting '
                    'n_warmup = n_iter for now.', RuntimeWarning)
                n_warmup = n_iter
            if self._trace.n_iter > self._trace.n_warmup and n_warmup > 0:
                warnings.warn(
                    self._prefix + 'self.trace indicates that warmup has '
                    'completed, so n_warmup will be set to 0.', RuntimeWarning)
                n_warmup = 0
            i_iter = self._trace.i_iter
            self._trace._n_iter += n_iter
            self._trace._n_warmup += n_warmup
            n_iter = self._trace._n_iter
            n_warmup = self._trace._n_warmup
            if verbose:
                n_run = n_iter - i_iter
                if n_update is None:
                    n_update = n_run // 10
                else:
                    n_update = int(n_update)
                    if n_update <= 0:
                        warnings.warn(
                            self._prefix + 'invalid n_update value. Using '
                            'n_run // 10 for now.', RuntimeWarning)
                        n_update = n_run // 10
                t_s = time.time()
                t_i = time.time()
            for i in range(i_iter, n_iter):
                if verbose:
                    if i > i_iter and not i % n_update:
                        t_d = time.time() - t_i
                        t_i = time.time()
                        n_div = np.sum(
                            self._trace._stats._diverging[-n_update:])
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
            return self.trace if return_copy else self._trace
        finally:
            warnings.showwarning = warnings._showwarning_orig
    
    @property
    def trace(self):
        return deepcopy(self._trace)
    
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
                raise ValueError('invalid value for dask_key.')
        self._dask_key = key
    
    @property
    def chain_id(self):
        return self._chain_id
    
    @chain_id.setter
    def chain_id(self, i):
        if i is None:
            i = random_str(6, '')
        else:
            try:
                i = str(i)
            except:
                raise ValueError('invalid value for chain_id.')
        self._chain_id = i
        self._prefix = ' CHAIN ' + self._chain_id + ' : '
