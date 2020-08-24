import numpy as np
from collections import namedtuple
from ..sample_trace import _HTrace
from .integration import CpuLeapfrogIntegrator, TCpuLeapfrogIntegrator
import warnings
from copy import deepcopy
import time
from multiprocess import Lock
try:
    from distributed import Pub
    HAS_DASK = True
except Exception:
    HAS_DASK = False

__all__ = ['BaseHMC']

# TODO: review the code


HMCStepData = namedtuple("HMCStepData",
                         "end, accept_stat, divergence_info, stats")


DivergenceInfo = namedtuple("DivergenceInfo", "message, exec_info, state")


class BaseHMC:
    """Base class to implement Hamiltonian Monte Carlo."""
    def __init__(self, logp_and_grad, sample_trace, dask_key=None,
                 process_lock=None):
        self._logp_and_grad = logp_and_grad
        self.dask_key = dask_key
        self.process_lock = process_lock
        if isinstance(sample_trace, self._expected_trace):
            self._sample_trace = sample_trace
        else:
            raise ValueError('invalid type for sample_trace.')
        self._chain_id = sample_trace.chain_id
        self._prefix = ' CHAIN #' + str(self._chain_id) + ' : '
        self.integrator = CpuLeapfrogIntegrator(self.sample_trace.metric,
                                                logp_and_grad)
        try:
            logp_0, grad_0 = logp_and_grad(self._sample_trace.x_0)
            assert np.isfinite(logp_0).all() and np.isfinite(grad_0).all()
        except Exception:
            raise ValueError('failed to get finite logp and/or grad at x_0.')
    
    '''
    # I have to remove this for now to make the multi inheritance
    # in THMC/TNUTS work correctly
    
    _expected_trace = _HTrace
    
    def _hamiltonian_step(self, start, p0, step_size):
        """Compute one hamiltonian trajectory and return the next state.

        Subclasses must overwrite this method and return a `HMCStepData`.
        """
        raise NotImplementedError("Abstract method")
    '''

    def astep(self):
        """Perform a single HMC iteration."""
        try:
            q0 = self._sample_trace._samples[-1]
        except Exception:
            q0 = self._sample_trace.x_0
            assert q0.ndim == 1
        p0 = self.sample_trace.metric.random(self.sample_trace.random_generator)
        start = self.integrator.compute_state(q0, p0)
        
        if not np.isfinite(start.energy):
            self.sample_trace.metric.raise_ok()
            raise RuntimeError(
                "Bad initial energy, please check the Hamiltonian at p = {}, "
                "q = {}.".format(p0, q0))
        
        step_size = self.sample_trace.step_size.current(self.warmup)
        hmc_step = self._hamiltonian_step(start, p0, step_size)
        self.sample_trace.step_size.update(hmc_step.accept_stat, self.warmup)
        self.sample_trace.metric.update(hmc_step.end.q, self.warmup)
        step_stats = self._expected_stats(
            **hmc_step.stats, **self.sample_trace.step_size.sizes(),
            warmup=self.warmup, diverging=bool(hmc_step.divergence_info))
        self.sample_trace.update(hmc_step.end.q, step_stats)
    
    def run(self, n_run=None, verbose=True, n_update=None):
        if self._dask_key is None:
            def sw(message, *args, **kwargs):
                warnings._showwarning_orig(self._prefix + str(message), *args,
                                           **kwargs)
        else:
            pub = Pub(self._dask_key)
            def sw(message, category, *args, **kwargs):
                pub.put([category, self._prefix + str(message)])
        try:
            warnings.showwarning = sw
            i_iter = self._sample_trace.i_iter
            n_iter = self._sample_trace.n_iter
            n_warmup = self._sample_trace.n_warmup
            if n_run is None:
                n_run = n_iter - i_iter
            else:
                try:
                    n_run = int(n_run)
                    assert n_run > 0
                except Exception:
                    raise ValueError(self._prefix + 'invalid value for n_run.')
                if n_run > n_iter - i_iter:
                    self._sample_trace.n_iter = i_iter + n_run
                    n_iter = self._sample_trace.n_iter
            if verbose:
                if n_update is None:
                    n_update = n_run // 5
                else:
                    try:
                        n_update = int(n_update)
                        assert n_update > 0
                    except Exception:
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
                        n_div = np.sum(
                            self._sample_trace.stats._diverging[-n_update:])
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
                            if self.has_lock:
                                self.process_lock.acquire()
                            print(msg_0 + msg_1 + msg_2)
                            if self.has_lock:
                                self.process_lock.release()
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
                    if self.has_lock:
                        self.process_lock.acquire()
                    print(msg)
                    if self.has_lock:
                        self.process_lock.release()
                else:
                    pub.put(['SamplingFinished', msg])
            return self._sample_trace
        finally:
            warnings.showwarning = warnings._showwarning_orig
    
    @property
    def sample_trace(self):
        return self._sample_trace
    
    @property
    def dask_key(self):
        return self._dask_key
    
    @dask_key.setter
    def dask_key(self, key):
        if key is None:
            pass
        else:
            if HAS_DASK:
                try:
                    key = str(key)
                except Exception:
                    raise ValueError('invalid value for dask_key.')
            else:
                raise RuntimeError('you give me the dask_key but have not '
                                   'installed dask.')
        self._dask_key = key
    
    @property
    def use_dask(self):
        return (self.dask_key is not None)
    
    @property
    def process_lock(self):
        return self._process_lock
    
    def process_lock(self, lock):
        if lock is None or isinstance(lock, Lock):
            self._process_lock = lock
        else:
            raise ValueError('invalid value for process_lock.')
    
    @property
    def has_lock(self):
        return (self.process_lock is not None)
    
    @property
    def chain_id(self):
        return self._chain_id


class BaseTHMC(BaseHMC):
    """Base class to implement Tempered HMC."""
    def __init__(self, logp_and_grad, sample_trace, dask_key=None,
                 process_lock=None):
        super().__init__(logp_and_grad, sample_trace, dask_key, process_lock)
        self.integrator = TCpuLeapfrogIntegrator(
            self.sample_trace.metric, logp_and_grad, self._logp_and_grad_base)
    
    def _logp_and_grad_base(self, x):
        logp, grad = self.sample_trace.density_base.logp_and_grad(
            x, original_space=False)
        return logp + self.sample_trace.logxi, grad
    
    def astep(self):
        """Perform a single HMC iteration."""
        try:
            q0 = self.sample_trace._samples[-1]
            u0 = self.sample_trace.stats._u[-1]
            Q0 = np.append(u0, q0)
        except Exception:
            q0 = self.sample_trace.x_0
            assert q0.ndim == 1
            u0 = np.random.normal(0,1)
            Q0 = np.append(u0, q0)
        p0 = self.sample_trace.metric.random(self.sample_trace.random_generator)
        v0 = self.sample_trace.random_generator.normal(0, 1)
        P0 = np.append(v0, p0)
        start = self.integrator.compute_state(Q0, P0)
        
        if not np.isfinite(start.energy):
            self.sample_trace.metric.raise_ok()
            raise RuntimeError(
                "Bad initial energy, please check the Hamiltonian at p = {}, "
                "q = {}, u = {}, v = {}, ".format(p0, q0, u0, v0))
        
        step_size = self.sample_trace.step_size.current(self.warmup)
        hmc_step = self._hamiltonian_step(start, P0, step_size)
        self.sample_trace.step_size.update(hmc_step.accept_stat, self.warmup)
        self.sample_trace.metric.update(hmc_step.end.q, self.warmup)
        step_stats = self._expected_stats(
            **hmc_step.stats, **self.sample_trace.step_size.sizes(),
            warmup=self.warmup, diverging=bool(hmc_step.divergence_info))
        self.sample_trace.update(hmc_step.end.q, step_stats)
    
    # _expected_trace = _TTrace
