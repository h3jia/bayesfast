import numpy as np
from scipy.special import logsumexp
import warnings

__all__ = ['importance']


def importance(logp_xq, logq_xq):
    try:
        lpq = np.asarray(logp_xq)
        lqq = np.asarray(logq_xq)
    except:
        raise ValueError('invalid value for the input.')
    
    if (lqq.ndim != 1) and (lqq.ndim != 2):
        raise ValueError(
            'dim of logq_xq should be 1 or 2, instead of {}.'.format(lqq.ndim))
    if lpq.shape != lqq.shape:
        raise ValueError('shape of logp_xq, {}, is different from shape of '
                         'logq_xq, {}.'.format(lpq.shape, lqq.shape))
    
    n_q = lqq.size
    lpqf = lpq.flatten()
    lqqf = lqq.flatten()
    
    logr = logsumexp(lpqf - lqqf, b=1 / n_q)
    foo = np.exp(lpqf - lqqf - logr)
    logr_err = (np.var(foo) / np.mean(foo)**2 / n_q)**0.5
    
    if logr_err > 0.25:
        warnings.warn('the estimated error for logr may be unreliable, '
                      'since the result is larger than 0.25.', RuntimeWarning)
    return logr, logr_err
