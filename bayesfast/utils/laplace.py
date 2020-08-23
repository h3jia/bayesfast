import numpy as np
import warnings
from collections import namedtuple
from numdifftools import Gradient, Hessian, Jacobian
from scipy.optimize import minimize
from scipy.linalg import sqrtm
from .sobol import multivariate_normal
from .misc import make_positive

__all__ = ['Laplace', 'untemper_laplace_samples']


LaplaceResult = namedtuple("LaplaceResult", 
                           "x_max, f_max, samples, cov, beta, opt_result")


class Laplace:
    """
    Evaluating and sampling the Laplace approximation for the target density.
    
    Parameters
    ----------
    optimize_method : str or callable, optional
        The `method` parameter for `scipy.optimize.minimize`. Set to
        `'Newton-CG'` by default.
    optimize_tol : float, optional
        The `tol` parameter for `scipy.optimize.minimize`. Set to `1e-5` by
        default.
    optimize_options : dict, optional
        The `options` parameter for `scipy.optimize.minimize`. Set to `{}` by
        default.
    max_cond : positive float, optional
        The maximum conditional number allowed for the Hessian matrix. All the
        eigenvalues that are smaller than `max_eigen_value / max_cond` will be
        truncated at this value. Set to `1e5` by default.
    n_sample : positive int or None, optional
        The number of samples to draw from the approximated Gaussian
        distribution. If `None`, will be determined by
        `min(1000, x_0.shape[-1] * 10)` during runtime. Set to `None` by
        default.
    beta : positive float, optional
        Scaling the approximate distribution `logq`, i.e. the final samples will
        come from `beta * logq`. Set to `1.` by default.
    mvn_generator : None or callable, optional
        Random number generator for the multivairate normal distribution. Should
        have signature `(mean, cov, size) -> samples`. If `None`, will use
        `bayesfast.utils.sobol.multivariate_normal`. Set to `None` by default.
    grad_options : dict, optional
        Additional keyword arguments for `numdifftools` to compute the gradient.
        Will be ignored if direct expression for the gradient is provided in
        `run`. Set to `{}` by default.
    hess_options : dict, optional
        Additional keyword arguments for `numdifftools` to compute the Hessian.
        Will be ignored if direct expression for the Hessian is provided in
        `run`. Set to `{}` by default.
    """
    def __init__(self, optimize_method='Newton-CG', optimize_tol=1e-5,
                 optimize_options={}, max_cond=1e5, n_sample=2000, beta=1.,
                 mvn_generator=None, grad_options={}, hess_options={}):
        if callable(optimize_method):
            self._optimize_method = optimize_method
        else:
            try:
                self._optimize_method = str(optimize_method)
            except Exception:
                raise ValueError('invalid value for optimize_method.')
        
        if optimize_tol is None:
            pass
        else:
            try:
                optimize_tol = float(optimize_tol)
                assert optimize_tol > 0
            except Exception:
                raise ValueError('invalid value for optimize_tol.')
        self._optimize_tol = optimize_tol
        
        try:
            self._optimize_options = dict(optimize_options)
        except Exception:
            raise ValueError('invalid value for optimize_options.')
        
        try:
            max_cond = float(max_cond)
            assert max_cond > 0
            self._max_cond = max_cond
        except Exception:
            raise ValueError('max_cond should be a positive float.')
        
        if n_sample is None:
            pass
        else:
            try:
                n_sample = int(n_sample)
                assert n_sample > 0
            except Exception:
                raise ValueError('invalid value for n_sample.')
        self._n_sample = n_sample
        
        try:
            beta = float(beta)
            assert beta > 0
            self._beta = beta
        except Exception:
            raise ValueError('beta should be a positive float.')
        
        if mvn_generator is None:
            mvn_generator = multivariate_normal
        if callable(mvn_generator):
            self._mvn_generator = mvn_generator
        else:
            raise ValueError('invalid value for mvn_generator.')
        
        try:
            self._grad_options = dict(grad_options)
        except Exception:
            raise ValueError('invalid value for grad_options.')
        
        try:
            self._hess_options = dict(hess_options)
        except Exception:
            raise ValueError('invalid value for hess_options.')
    
    def run(self, logp, x_0=None, grad=None, hess=None):
        if not callable(logp):
            raise ValueError('logp should be callable.')
        try:
            x_0 = np.atleast_1d(x_0)
            assert x_0.ndim == 1
        except Exception:
            raise ValueError('invalid value for x_0.')
        if self._n_sample is None:
            n_sample = min(1000, x_0.shape[-1] * 10)
        else:
            n_sample = self._n_sample
        if not callable(hess):
            if callable(grad):
                def _hess(*args, **kwargs):
                    foo = Jacobian(grad, **self._hess_options)(*args, **kwargs)
                    return (foo + foo.T) / 2
                hess = _hess
            else:
                hess = Hessian(logp, **self._hess_options)
        if not callable(grad):
            grad = Gradient(logp, **self._grad_options)
        
        opt = minimize(fun=lambda x: -logp(x), x0=x_0,
                       method=self._optimize_method, jac=lambda x: -grad(x),
                       hess=lambda x: -hess(x), tol=self._optimize_tol,
                       options=self._optimize_options)
        if not opt.success:
            warnings.warn(
                'the optimization stopped at {}, but maybe it has not '
                'converged yet.'.format(opt.x), RuntimeWarning)
        x_max = opt.x
        f_max = -opt.fun
        cov = np.linalg.inv(make_positive(-hess(x_max), self._max_cond))
        samples = self._mvn_generator(x_max, cov / self._beta, n_sample)
        return LaplaceResult(x_max, f_max, samples, cov, self._beta, opt)


def untemper_laplace_samples(laplace_result):
    if isinstance(laplace_result, LaplaceResult):
        delta = laplace_result.samples - laplace_result.x_max
        delta *= laplace_result.beta**0.5
        return laplace_result.x_max + delta
    else:
        raise ValueError('laplace_result should be a LaplaceResult.')
