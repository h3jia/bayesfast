import numpy as np
import warnings
from collections import namedtuple
from numdifftools import Gradient, Hessian
from scipy.optimize import minimize
from scipy.linalg import sqrtm
from ..utils.random_utils import random_multivariate_normal

__all__ = ['Laplace']


# TODO: random_state
LaplaceResult = namedtuple("LaplaceResult", "x_max, logp_max, samples")

class Laplace:
    
    def __init__(self, logp, x_0=None, x_max=None, logp_max=None, grad=None, 
                 hess=None, logp_args=()):
        if not callable(logp):
            raise ValueError('logp should be callable.')
        if x_max is None:
            self._x_0 = np.atleast_1d(x_0)
            if self._x_0.ndim != 1:
                raise ValueError('x_0 should be a 1-d array.')
            self._x_max = None
            self._logp_max = None
        else:
            self._x_0 = None
            self._x_max = np.atleast_1d(x_max)
            if self._x_max.ndim != 1:
                raise ValueError('x_max should be a 1-d array.')
            self._logp_max = float(logp_max) if (logp_max is not None) else None
        self._logp = logp
        self._grad = grad if callable(grad) else Gradient(logp)
        self._hess = hess if callable(hess) else Hessian(logp)
        self._logp_args = logp_args
        
    def run(self, n_sample, beta=1, optimize_method='Newton-CG', 
            optimize_options={}):
        if self._x_max is None:
            opt = minimize(lambda x: -self._logp(x), self._x_0, self._logp_args, 
                           optimize_method, lambda x: -self._grad(x), 
                           lambda x: -self._hess(x), options=optimize_options)
            if not opt.success:
                warnings.warn(
                    'the optimization stopped at {}, but probably it has not '
                    'converged yet.'.format(opt.x), RuntimeWarning)
            self._x_max = opt.x
            self._logp_max = -opt.fun
        if self._logp_max is None:
            self._logp_max = self._logp(self._x_max)
        cov = -np.linalg.inv(self._hess(self._x_max))
        if not np.all(np.linalg.eigvalsh(cov) > 0):
            warnings.warn(
                'not all the eigenvalues of the Hessian at MAP are positive. '
                'I will square it and then take the square root for now.')
            cov = sqrtm(cov @ cov)
        n_sample = int(n_sample)
        if n_sample <= 0:
            raise ValueError('n_sample should be a positive int.')
        beta = float(beta)
        if beta <= 0:
            raise ValueError('beta should be a positive float.')
        samples = random_multivariate_normal(self._x_max, beta * cov, n_sample)
        return LaplaceResult(self._x_max, self._logp_max, samples)
