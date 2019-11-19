import numpy as np
import warnings
from collections import namedtuple
from numdifftools import Gradient, Hessian, Jacobian
from scipy.optimize import minimize
from scipy.linalg import sqrtm
from ..utils.random import multivariate_normal

__all__ = ['Laplace']


# TODO: random_state
LaplaceResult = namedtuple("LaplaceResult", "x_max, f_max, samples, opt_result")


def _make_positive(A, max_cond=10000.):
    a, w = np.linalg.eigh(A)
    if a[-1] <= 0:
        raise ValueError('all the eigenvalues are non-positive.')
    i = np.argmax((a - a[-1] / max_cond) > 0)
    a[:i] = a[i]
    return w @ np.diag(a) @ w.T


class Laplace:
    
    def __init__(self, logp, x_0=None, x_max=None, f_max=None, grad=None, 
                 hess=None, logp_args=()):
        if not callable(logp):
            raise ValueError('logp should be callable.')
        if x_max is None:
            self._x_0 = np.atleast_1d(x_0)
            if self._x_0.ndim != 1:
                raise ValueError('x_0 should be a 1-d array.')
            self._x_max = None
            self._f_max = None
        else:
            self._x_0 = None
            self._x_max = np.atleast_1d(x_max)
            if self._x_max.ndim != 1:
                raise ValueError('x_max should be a 1-d array.')
            self._f_max = float(f_max) if (f_max is not None) else None
        self._logp = logp
        self._grad = grad if callable(grad) else Gradient(logp)
        if callable(hess):
            self._hess = hess
        elif callable(grad):
            def _hess(*args, **kwargs):
                foo = Jacobian(grad)(*args, **kwargs)
                return (foo + foo.T) / 2
            self._hess = _hess
        else:
            self._hess = Hessian(logp)
        self._logp_args = logp_args
        
    def run(self, n_sample=2000, beta=1, optimize_method='Newton-CG', 
            optimize_options={}, max_cond=1e5):
        n_sample = int(n_sample)
        if n_sample <= 0:
            raise ValueError('n_sample should be a positive int.')
        beta = float(beta)
        if beta <= 0:
            raise ValueError('beta should be a positive float.')
        max_cond = float(max_cond)
        if max_cond <= 0:
            raise ValueError('max_cond should be a positive float.')
        
        if self._x_max is None:
            opt = minimize(lambda x: -self._logp(x), self._x_0, self._logp_args, 
                           optimize_method, lambda x: -self._grad(x), 
                           lambda x: -self._hess(x), options=optimize_options)
            if not opt.success:
                warnings.warn(
                    'the optimization stopped at {}, but probably it has not '
                    'converged yet.'.format(opt.x), RuntimeWarning)
            self._x_max = opt.x
            self._f_max = -opt.fun
        else:
            opt = None
        if self._f_max is None:
            self._f_max = self._logp(self._x_max)
        cov = np.linalg.inv(_make_positive(-self._hess(self._x_max), max_cond))
        samples = multivariate_normal(self._x_max, beta * cov, n_sample)
        return LaplaceResult(self._x_max, self._f_max, samples, opt)
