import numpy as np
from ..utils.vectorize import vectorize
from scipy.special import logsumexp
from scipy.optimize import root_scalar
from scipy.stats import multivariate_normal
from ..utils import integrated_time
from ..utils.random import check_state
import warnings

#__all__ = ['BS', 'WarpBS']
__all__ = ['BS', 'adapt_nq']


def BS(logp_xp, logp_xq, logq_xp, logq_xq):
    lpp = np.asarray(logp_xp)
    lpq = np.asarray(logp_xq)
    lqp = np.asarray(logq_xp)
    lqq = np.asarray(logq_xq)
    
    if (lqq.ndim != 1) and (lqq.ndim != 2):
        raise ValueError(
            'dim of logq_xq should be 1 or 2, instead of {}.'.format(lqq.ndim))
    if (lpp.ndim != 1) and (lpp.ndim != 2):
        raise ValueError(
            'dim of logp_xp should be 1 or 2, instead of {}.'.format(lpp.ndim))
    if lpp.shape != lqp.shape:
        raise ValueError('shape of logp_xp, {}, is different from shape of '
                         'logq_xp, {}.'.format(lpp.shape, lqp.shape))
    if lpq.shape != lqq.shape:
        raise ValueError('shape of logp_xq, {}, is different from shape of '
                         'logq_xq, {}.'.format(lpq.shape, lqq.shape))
    
    n_p = lpp.size
    n_q = lqq.size
    lppf = lpp.flatten()
    lpqf = lpq.flatten()
    lqpf = lqp.flatten()
    lqqf = lqq.flatten()
    
    _a = lqpf - lppf - np.log(n_p / n_q)
    _b = lpqf - lqqf + np.log(n_p / n_q)
    
    def score(logr):
        _c = logsumexp(logr + _a - logsumexp(np.array((logr + _a, 
                       np.zeros_like(_a))), axis=0))
        _d = logsumexp(-logr + _b - logsumexp(np.array((-logr + _b, 
                       np.zeros_like(_b))), axis=0))
        return _c - _d
    
    logr = root_scalar(score, x0=0., x1=5.).root
    _f1 = np.exp(lpqf - logr - logsumexp(np.array((lpqf - logr + 
                 np.log(n_p / (n_p + n_q)), lqqf + np.log(n_q / (n_p + n_q)))), 
                 axis=0))
    _f2 = np.exp(lqpf - logsumexp(np.array((lppf - logr + 
                 np.log(n_p / (n_p + n_q)), lqpf + np.log(n_q / (n_p + n_q)))), 
                 axis=0))
    tau = integrated_time(_f2.reshape(lpp.shape)[..., np.newaxis])[0]
    re2_p = tau * np.var(_f2) / np.mean(_f2)**2 / n_p
    re2_q = np.var(_f1) / np.mean(_f1)**2 / n_q
    logr_err = (re2_p + re2_q)**0.5
    
    if logr_err > 0.5:
        warnings.warn('the estimated error for logr may be unreliable since it '
                      'is larger than 0.5.')
    return logr, logr_err, re2_p, re2_q


def adapt_nq(re2_p, re2_q, n_q0, n_call_p, f_err=0.1, f_eva=0.1):
    try:
        f_err = float(f_err)
        f_eva = float(f_eva)
        assert (0. < f_err < 1.) and (0. < f_eva < 1.)
    except:
        raise ValueError('invalid values for f_err and/or f_eva.')
    from_err = n_q0 * (1 - f_err) * re2_q / f_err / re2_p
    from_eva = n_call_p * f_eva / (1 - f_eva)
    return int(max(n_q0, min(from_err, from_eva)))


"""
NOTE: below is some legacy code that is not compatible with current BayesFast
      we will revise it later
"""

#TODO: enable reusing logp_xp in WarpBS

def WarpBS(xp, logp, n_q=None, logp_xp=None, logp_vectorized=True, 
           logp_vectorize_level=1, random_state=None):
    if logp_xp is not None:
        raise NotImplementedError(
            'reusing logp_xp values is not implemented yet.')
    if not logp_vectorized:
        logp = vectorize(logp, logp_vectorize_level)
    xp = np.asarray(xp)
    if xp.ndim != 2 and xp.ndim != 3:
        raise NotImplementedError('xp should be a 2-d or 3-d array for now.')
    if xp.shape[0] == 1:
        if xp.shape[1] == 1:
            raise ValueError('it seems that xp has only one point.')
        else:
            xp = xp[0]
    _n = xp.shape[0] // 2
    xp_0 = xp[:_n].reshape((-1, xp.shape[-1]))
    xp_1 = xp[_n:(2 * _n)]
    xp_mu = np.mean(xp_0, axis=0)
    xp_cov = np.cov(xp_0, rowvar=False)
    rs = check_state(random_state)
    xq = rs.multivariate_normal(xp_mu, xp_cov, 
                                xp_0.shape[0] if n_q is None else n_q)
    logps = lambda x: (np.logaddexp(logp(x), logp(2 * xp_mu - x)) - np.log(2.))
    logq = lambda x: multivariate_normal.logpdf(x, xp_mu, xp_cov)
    return BS(xp_1, xq, logps, logq)
