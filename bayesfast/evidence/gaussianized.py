import numpy as np
from .bridge import bridge
from .importance import importance
from .harmonic import harmonic
from ..utils.parallel import ParallelBackend, get_backend
from ..transforms import SIT
from ..samplers import TraceTuple
from threadpoolctl import threadpool_limits
import warnings

__all__ = ['GBS', 'GIS', 'GHM']

# TODO: directly get logp_p from TraceTuple


a = """
    Gaussianized {} for normalizing constant estimation.
    
    Parameters
    ----------
    sit : SIT, dict or None, optional
        The `SIT` generative model.
    parallel_backend : None, int, Pool, Client or ParallelBackend, optional
        The backend for parallelization. If `None`, will use the bayesfast
        global parallel backend. Otherwise, will be passed to initialize
        a ParallelBackend.
    """


b = """n_q : positive int or None, optional
        The number of samples to draw from the SIT model.
    f_call : positive float or None
        Another way to control `n_q`.
    """


class _GBase:
    """Utilities shared by GBS, GIS and GHM."""
    def __init__(self, sit=None, parallel_backend=None):
        self.sit = sit
        self.parallel_backend = parallel_backend
    
    @property
    def sit(self):
        return self._sit
    
    @sit.setter
    def sit(self, s):
        if s is None:
            s = {}
        if isinstance(s, dict):
            s = SIT(**s)
        elif isinstance(s, SIT):
            pass
        else:
            raise ValueError('invalid value for sit.')
        self._sit = s
    
    @property
    def parallel_backend(self):
        if self._parallel_backend is None:
            return get_backend()
        else:
            return self._parallel_backend
    
    @parallel_backend.setter
    def parallel_backend(self, backend):
        if backend is None:
            self._parallel_backend = None
        else:
            self._parallel_backend = ParallelBackend(backend)
    
    def run(self, x_p, logp, logp_p=None):
        raise NotImplementedError('abstrace method.')
    
    __call__ = run


class _GBaseQ(_GBase):
    """Utilities shared by GBS and GIS."""
    def __init__(self, sit=None, parallel_backend=None, n_q=None, f_call=0.05):
        super().__init__(sit, parallel_backend)
        self.n_q = n_q
        self.f_call = f_call
    
    @property
    def n_q(self):
        return self._n_q
    
    @n_q.setter
    def n_q(self, n):
        if n is None:
            pass
        else:
            try:
                n = int(n)
                assert n > 0
            except Exception:
                raise ValueError('invalid value for n_q.')
        self._n_q = n
    
    @property
    def f_call(self):
        return self._f_call
    
    @f_call.setter
    def f_call(self, f):
        if f is None:
            pass
        else:
            try:
                f = float(f)
                assert f > 0
            except Exception:
                raise ValueError('invalid value for f_call.')
        self._f_call = f
    
    def run(self, x_p, logp, logp_p=None):
        if not callable(logp):
            raise ValueError('logp should be callable.')
        if isinstance(x_p, TraceTuple):
            pass
        else:
            try:
                x_p = np.asarray(x_p)
                assert 2 <= x_p.ndim <= 3
            except Exception:
                raise ValueError('invalid value for x_p.')
        
        if self.n_q is not None:
            n_q = self.n_q
            if isinstance(x_p, TraceTuple):
                x_p = x_p.get(flatten=False)
        else:
            f_call = self.f_call
            if f_call is not None:
                if isinstance(x_p, TraceTuple):
                    n_p = x_p.n_call
                    n_q = int(n_p * f_call)
                    x_p = x_p.get(flatten=False)
                elif isinstance(x_p, np.ndarray):
                    warnings.warn('f_call should be used only when x_p is a '
                                  'TraceTuple. Using equal-sample allocation '
                                  'for now.', RuntimeWarning)
                    f_call = None
                else:
                    raise RuntimeError('unexpected value for x_p.')
            if f_call is None:
                if isinstance(x_p, TraceTuple):
                    x_p = x_p.get(flatten=False)
                if isinstance(x_p, np.ndarray):
                    n_q = np.prod(x_p.shape[:-1])
                else:
                    raise RuntimeError('unexpected value for x_p.')
        
        try:
            assert x_p.shape[-1] > 1
            assert np.prod(x_p.shape[:-1]) > 1
        except Exception:
            raise ValueError('invalid shape for x_p.')
        if x_p.shape[0] == 1:
            x_p = x_p[0] # the case for one single chain
        
        return self._compute_evidence(logp, x_p, logp_p, n_q)
    
    __call__ = run
    
    def _compute_evidence(self, logp, x_p, logp_p, n_q):
        raise NotImplementedError('abstract method.')
    
    def _map(self, logp, x):
        with self.parallel_backend:
            x_shape = x.shape
            x = x.reshape((-1, x_shape[-1]))
            map_result = self.parallel_backend.map(logp, x)
        return np.asarray(map_result).reshape(x_shape[:-1])


class GBS(_GBaseQ):
    
    __doc__ = (a + b).format('Bridge Sampling')
    
    def _compute_evidence(self, logp, x_p, logp_p, n_q):
        n_half = x_p.shape[0] // 2
        self.sit.fit(data=x_p[:n_half])
        x_q = self.sit.sample(n_q)[0]
        
        if logp_p is not None:
            try:
                logp_p = np.asarray(logp_p)
                assert logp_p.shape == x_p.shape[:-1]
                logp_p = logp_p[n_half:]
            except Exception:
                warnings.warn(
                    'the logp_p you gave me seems not correct. Will recompute '
                    'it from logp and x_p.', RuntimeWarning)
                logp_p = None
        
        if logp_p is None:
            logp_p = self._map(logp, x_p[n_half:])
            if logp_p.shape != x_p[n_half:].shape[:-1]:
                raise RuntimeError(
                    'the shape of logp_p, {}, does not match the shape of '
                    '(the second half of) x_p, {}.'.format(logp_p.shape,
                    x_p[n_half:].shape))
        
        logp_q = self._map(logp, x_q)
        if logp_q.shape != x_q.shape[:-1]:
            raise RuntimeError(
                'the shape of logp_q, {}, does not match the shape of x_q, '
                '{}.'.format(logp_q.shape, x_q.shape))
        
        logq_p = self.sit.logq(x_p[n_half:])
        logq_q = self.sit.logq(x_q)
        return bridge(logp_p, logp_q, logq_p, logq_q)


class GIS(_GBaseQ):
    
    __doc__ = (a + b).format('Importance Sampling')
    
    def _compute_evidence(self, logp, x_p, logp_p, n_q):
        self.sit.fit(data=x_p)
        x_q = self.sit.sample(n_q)[0]
        
        logp_q = self._map(logp, x_q)
        if logp_q.shape != x_q.shape[:-1]:
            raise RuntimeError(
                'the shape of logp_q, {}, does not match the shape of x_q, '
                '{}.'.format(logp_q.shape, x_q.shape))
        
        logq_q = self.sit.logq(x_q)
        return importance(logp_q, logq_q)


class GHM(_GBase):
    
    __doc__ = a.format('Harmonic Mean')
    
    def run(self, x_p, logp=None, logp_p=None):
        if isinstance(x_p, TraceTuple):
            x_p = x_p.get(flatten=False)
        else:
            try:
                x_p = np.asarray(x_p)
                assert 2 <= x_p.ndim <= 3
            except Exception:
                raise ValueError('invalid value for x_p.')
        
        try:
            assert x_p.shape[-1] > 1
            assert np.prod(x_p.shape[:-1]) > 1
        except Exception:
            raise ValueError('invalid shape for x_p.')
        if x_p.shape[0] == 1:
            x_p = x_p[0] # the case for one single chain
        
        n_half = x_p.shape[0] // 2
        
        if logp_p is not None:
            try:
                logp_p = np.asarray(logp_p)
                assert logp_p.shape == x_p.shape[:-1]
                logp_p = logp_p[n_half:]
            except Exception:
                warnings.warn(
                    'the logp_p you gave me seems not correct. Will recompute '
                    'it from logp and x_p.', RuntimeWarning)
                logp_p = None
        
        if logp_p is None:
            if not callable(logp):
                raise ValueError('you gave me neither the correct logp_p nor a '
                                 'callable logp function.')
            logp_p = self._map(logp, x_p[n_half:])
            if logp_p.shape != x_p[n_half:].shape[:-1]:
                raise RuntimeError(
                    'the shape of logp_p, {}, does not match the shape of '
                    '(the second half of) x_p, {}.'.format(logp_p.shape,
                    x_p[n_half:].shape))
        
        self.sit.fit(data=x_p[:n_half])
        logq_p = self.sit.logq(x_p[n_half:])
        return harmonic(logp_p, logq_p)
    
    __call__ = run
