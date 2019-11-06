import numpy as np
from scipy.special import expit
import multiprocessing as mp
import warnings
import time

from ..samplers.pymc3.nuts import NUTS
from ..utils.warnings import SamplingProgess
from ..utils.random_utils import check_state, split_state

__all__ = ['get_sig_beta', 'TemperedDensity', 'AIS']


# TODO: add support for logp_and_grad, probably use Density class
# TODO: add support for lambda functions with pool
# TODO: improve progress output
# TODO: improve checks in AIS


def get_sig_beta(n, delta=4):
    n = int(n)
    if not n >= 2:
        raise ValueError('n should be at least 2, instead of {}.'.format(n))
    _beta = expit(delta * (2 * np.arange(n) / (n - 1) - 1))
    beta = (_beta - _beta[0]) / (_beta[-1] - _beta[0])
    return beta


class TemperedDensity:
    
    def __init__(self, logp, grad, logp_0, grad_0, beta, start=0):
        self._logp = logp
        self._logp_0 = logp_0
        self._grad = grad
        self._grad_0 = grad_0
        if isinstance(beta, int):
            self._n = beta
            self._beta = get_sig_beta(self._n)
        else:
            try:
                beta = np.asarray(beta)
                beta = beta.reshape(-1)
                assert beta.size >= 2
            except:
                raise ValueError('beta should be an array with len >= 2.')
            self._n = beta.size
            self._beta = beta
        self.set(start)
    
    def _get_b(self, i):
        if i is not None:
            self.set(i)
        return self._beta[self._i]
    
    def logp_t(self, x=None, i=None, cached=False):
        b = self._get_b(i)
        if cached and x is None:
            pass
        else:
            if self._i > 0:
                self._logp_c = self._logp(x)
            if self._i < self._n - 1:
                self._logp_0_c = self._logp_0(x)
        if self._i == 0:
            return self._logp_0_c
        elif self._i == self._n - 1:
            return self._logp_c
        else:
            return b * self._logp_c + (1 - b) * self._logp_0_c
    
    def grad_t(self, x=None, i=None, cached=False):
        b = self._get_b(i)
        if cached and x is None:
            pass
        else:
            if self._i > 0:
                self._grad_c = self._grad(x)
            if self._i < self._n - 1:
                self._grad_0_c = self._grad_0(x)
        if self._i == 0:
            return self._grad_0_c
        elif self._i == self._n - 1:
            return self._grad_c
        else:
            return b * self._grad_c + (1 - b) * self._grad_0_c
    
    def logp_and_grad_t(self, x=None, i=None, cached=False):
        b = self._get_b(i)
        if cached and x is None:
            pass
        else:
            if self._i > 0:
                self._logp_c = self._logp(x)
                self._grad_c = self._grad(x)
            if self._i < self._n - 1:
                self._logp_0_c = self._logp_0(x)
                self._grad_0_c = self._grad_0(x)
        if self._i == 0:
            return (self._logp_0_c, self._grad_0_c)
        elif self._i == self._n - 1:
            return (self._logp_c, self._grad_c)
        else:
            return (b * self._logp_c + (1 - b) * self._logp_0_c, 
                    b * self._grad_c + (1 - b) * self._grad_0_c)
    
    @property
    def i(self):
        return self._i
    
    @property
    def n(self):
        return self._n
    
    @property
    def beta(self):
        return self._beta.copy()
    
    def next(self):
        if self._i < self._n - 1:
            self._i += 1
        else:
            warnings.warn('we are already at the last beta.', RuntimeWarning)
    
    def previous(self):
        if self._i > 0:
            self._i -= 1
        else:
            warnings.warn('we are already at the first beta.', RuntimeWarning)
    
    def set(self, index):
        index = int(index)
        if not 0 <= index < self._n:
            start = 0
            warnings.warn(
                'index is out of range. Use 0 for now.', RuntimeWarning)
        self._i = int(index)
    
    def set_first(self):
        self._i = 0
    
    def set_last(self):
        self._i = self._n - 1

class _AIS:
    
    def __init__(self, logp_t, x_0, reverse, n_warmup, random_state, 
                 nuts_kwargs, verbose, n_update):
        self.logp_t = logp_t
        self.x_0 = x_0
        self.reverse = reverse
        self.n_warmup = n_warmup
        self.random_state = random_state
        self.nuts_kwargs = nuts_kwargs
        self.verbose = verbose
        self.n_update = n_update
    
    def worker(self, i):
        import os
        os.environ['OMP_NUM_THREADS'] = '1'
        import bayesfast.utils.warnings as bfwarnings
        import warnings
        warnings.simplefilter('always', SamplingProgess)
        warnings.showwarning = bfwarnings.showwarning_chain(i)
        warnings.formatwarning = bfwarnings.formatwarning_chain(i)
        self.logp_t.set(self.logp_t._n // 2)
        nuts = NUTS(logp_and_grad=self.logp_t.logp_and_grad_t, x_0=self.x_0[i], 
                    random_state=self.random_state[i], **self.nuts_kwargs)
        logz = 0
        
        t_s = time.time()
        t_i = time.time()
        _w0 = int(np.floor(self.n_warmup / 2.))
        tt = nuts.run(_w0, _w0, verbose=False, return_copy=False)
        if self.reverse:
            self.logp_t.set_last()
        else:
            self.logp_t.set_first()
        _w1 = int(np.ceil(self.n_warmup / 2.))
        _wc = tt.n_call
        tt = nuts.run(_w1, 0, verbose=False, return_copy=False)
        _wc = tt.n_call - _wc
        t_d = time.time() - t_i
        t_i = time.time()
        warnings.warn(
            'sampling initializing [ 0 / {} ], executed {} warmup iterations '
            'in {:.2f} seconds.'.format(self.logp_t._n, self.n_warmup, t_d), 
            SamplingProgess)
        
        logz -= tt._stats._logp[-1]
        if self.reverse:
            self.logp_t.previous()
        else:
            self.logp_t.next()
        logz += self.logp_t.logp_t(tt._samples[-1])
        for j in range(1, self.logp_t._n - 1):
            if self.verbose and not j % self.n_update:
                t_d = time.time() - t_i
                t_i = time.time()
                warnings.warn(
                    'sampling proceeding [ {} / {} ], last {} iterations used '
                    '{:.2f} seconds.'.format(j, self.logp_t._n, self.n_update, 
                    t_d), SamplingProgess)
            tt = nuts.run(1, 0, verbose=False, return_copy=False)
            logz -= tt._stats._logp[-1]
            if self.reverse:
                self.logp_t.previous()
            else:
                self.logp_t.next()
            logz += self.logp_t.logp_t(tt._samples[-1])
        if self.verbose:
            t_f = time.time()
            warnings.warn(
                'sampling finished [ {} / {} ], executed {} iterations in '
                '{:.2f} seconds.'.format(self.logp_t._n, self.logp_t._n, 
                self.logp_t._n, t_f - t_s), SamplingProgess)
        if self.reverse:
            return -logz, tt.n_call + self.logp_t._n - 1, tt.n_call
        else:
            return logz, tt.n_call - _wc + self.logp_t._n - 1, tt.n_call - _wc
    
    __call__ = worker

        
def AIS(logp_t, x_0, pool=None, m_pool=None, reverse=False, n_warmup=500, 
        random_state=None, nuts_kwargs={}, verbose=True, n_update=None):
    if not isinstance(logp_t, TemperedDensity):
        raise ValueError('logp_t should be a TemperedDensity.')
    x_0 = np.asarray(x_0)
    if not x_0.ndim == 2:
        raise ValueError('x_0 should be a 2-d array.')
    m, n = x_0.shape
    if hasattr(random_state, '__iter__'):
        random_state = [check_state(_) for _ in random_state]
    else:
        random_state = split_state(check_state(random_state), m)
    nuts_kwargs = nuts_kwargs.copy()
    if 'logp_and_grad' in nuts_kwargs:
        del nuts_kwargs['logp_and_grad']
        warnings.warn('nuts_kwargs contains logp_and_grad, which will not be '
                      'used.', RuntimeWarning)
    if 'x_0' in nuts_kwargs:
        del nuts_kwargs['x_0']
        warnings.warn('nuts_kwargs contains x_0, which will not be used.', 
                      RuntimeWarning)
    if 'random_state' in nuts_kwargs:
        del nuts_kwargs['random_state']
        warnings.warn('nuts_kwargs contains random_state, which will not be '
                      'used.', RuntimeWarning)
    """if not 'metric' in nuts_kwargs:
        nuts_kwargs['metric'] = np.diag(np.cov(x_0, rowvar=False))"""
    _new_pool = False
    if pool is None:
        n_pool = min(m, m_pool) if (m_pool is not None) else m
        pool = mp.Pool(n_pool)
        _new_pool = True
    elif pool is False:
        pass
    else:
        if not hasattr(pool, 'map'):
            raise ValueError('pool does not have attribute "map".')
    if n_update is None:
        n_update = logp_t._n // 5
    worker = _AIS(logp_t, x_0, reverse, n_warmup, random_state, nuts_kwargs, 
                  verbose, n_update)
    if pool:
        map_result = pool.map(worker, np.arange(m))
    else:
        map_result = list(map(worker, np.arange(m)))
    if _new_pool:
        pool.close()
        pool.join()
    logz = [ms[0] for ms in map_result]
    n_logp = [ms[1] for ms in map_result]
    n_grad = [ms[2] for ms in map_result]
    if x_0.shape[0] == 1:
        # we cannot estimate the error with only one simulation
        return np.mean(logz), None, np.sum(n_logp), np.sum(n_grad)
    else:
        return (np.mean(logz), np.std(logz) / m**0.5, np.sum(n_logp), 
                np.sum(n_grad))
