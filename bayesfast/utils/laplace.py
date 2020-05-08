import numpy as np
import warnings
from collections import namedtuple
from numdifftools import Gradient, Hessian, Jacobian
from scipy.optimize import minimize
from scipy.linalg import sqrtm
from ..utils.random import multivariate_normal

__all__ = ['Laplace']


LaplaceResult = namedtuple("LaplaceResult", 
                           "x_max, f_max, samples, cov, opt_result")


def make_positive(A, max_cond=1e5):
    a, w = np.linalg.eigh(A) # a: all the eigenvalues, in ascending order
    if a[-1] <= 0:
        raise ValueError('all the eigenvalues are non-positive.')
    i = np.argmax(a > a[-1] / max_cond)
    a[:i] = a[i]
    return w @ np.diag(a) @ w.T


class Laplace:
    """
    Evaluating and sampling the Laplace approximation for the target density.
    
    Parameters
    ----------
    optimize_method : str or callable, optional
        The `method` parameter for `scipy.optimize.minimize`. Set to
        `'Newton-CG'` by default.
    optimize_options : dict, optional
        The `options` parameter for `scipy.optimize.minimize`. Set to `{}` by
        default.
    max_cond : positive float, optional
        The maximum conditional number allowed for the Hessian matrix. All the
        eigenvalues that are smaller than `max_eigen_value / max_cond` will be
        truncated at this value. Set to `1e5` by default.
    beta : positive float, optional
        Scaling the approximate distribution `logq`, i.e. the final samples will
        come from `beta * logq`. Set to `1.` by default.
    random_options : dict, optional
        Additional keyword arguments for `bf.utils.random.multivariate_normal`
        (other than `mean`, `cov` and `size`). Set to `{}` by default.
    grad_options : dict, optional
        Additional keyword arguments for `numdifftools` to compute the gradient.
        Will be ignored if direct expression for the gradient is provided in
        `run`. Set to `{}` by default.
    hess_options : dict, optional
        Additional keyword arguments for `numdifftools` to compute the Hessian.
        Will be ignored if direct expression for the Hessian is provided in
        `run`. Set to `{}` by default.
    """
    def __init__(self, optimize_method='Newton-CG', optimize_options={},
                 max_cond=1e5, n_sample=2000, beta=1., random_options={},
                 grad_options={}, hess_options={}):
        if callable(optimize_method):
            self._optimize_method = optimize_method
        else:
            try:
                self._optimize_method = str(optimize_method)
            except:
                raise ValueError('invalid value for optimize_method.')
        try:
            self._optimize_options = dict(optimize_options)
        except:
            raise ValueError('invalid value for optimize_options.')
        try:
            max_cond = float(max_cond)
            assert max_cond > 0
            self._max_cond = max_cond
        except:
            raise ValueError('max_cond should be a positive float.')
        try:
            n_sample = int(n_sample)
            assert n_sample > 0
            self._n_sample = n_sample
        except:
            raise ValueError('n_sample should be a positive int.')
        try:
            beta = float(beta)
            assert beta > 0
            self._beta = beta
        except:
            raise ValueError('beta should be a positive float.')
        try:
            self._random_options = dict(random_options)
        except:
            raise ValueError('invalid value for random_options.')
        try:
            self._grad_options = dict(grad_options)
        except:
            raise ValueError('invalid value for grad_options.')
        try:
            self._hess_options = dict(hess_options)
        except:
            raise ValueError('invalid value for hess_options.')
    
    def run(self, logp, x_0=None, grad=None, hess=None):
        if not callable(logp):
            raise ValueError('logp should be callable.')
        try:
            x_0 = np.atleast_1d(x_0)
            assert x_0.ndim == 1
        except:
            raise ValueError('invalid value for x_0.')
        if not callable(hess):
            if callable(grad):
                def _hess(*args, **kwargs):
                    foo = Jacobian(grad, **hess_options)(*args, **kwargs)
                    return (foo + foo.T) / 2
                hess = _hess
            else:
                hess = Hessian(logp, **hess_options)
        if not callable(grad):
            grad = Gradient(logp, **grad_options)
        
        opt = minimize(fun=lambda x: -logp(x), x0=x_0,
                       method=self._optimize_method, jac=lambda x: -grad(x),
                       hess=lambda x: -hess(x), options=self._optimize_options)
        if not opt.success:
            warnings.warn(
                'the optimization stopped at {}, but probably it has not '
                'converged yet.'.format(opt.x), RuntimeWarning)
        x_max = opt.x
        f_max = -opt.fun
        cov = np.linalg.inv(make_positive(-hess(x_max), self._max_cond))
        samples = multivariate_normal(x_max, cov / self._beta, self._n_sample,
                                      **self._random_options)
        return LaplaceResult(x_max, f_max, samples, cov, opt)
