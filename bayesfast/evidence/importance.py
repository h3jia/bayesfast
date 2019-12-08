"""
NOTE: below is some legacy code that is not compatible with current BayesFast
      we will revise it later
"""

import numpy as np
from ..utils.vectorize import vectorize
from scipy.special import logsumexp
import warnings

__all__ = ['IS']


def IS(xq=None, logp=None, logq=None, logp_xq=None, logq_xq=None, 
       logp_vectorized=True, logq_vectorized=True, logp_vectorize_level=1, 
       logq_vectorize_level=1):
    if not logp_vectorized:
        logp = vectorize(logp, logp_vectorize_level)
    if not logq_vectorized:
        logq = vectorize(logq, logq_vectorize_level)
    
    if logp_xq is not None:
        _lpq = logp_xq
    elif (xq is not None) and (logp is not None):
        try:
            _lpq = logp(xq)
        except:
            raise RuntimeError('failed to get logp_xq from logp and xq.')
    else:
        raise ValueError('cannot obtain logp_xq from what you gave me.')
    _lpq = np.asarray(_lpq)
    if (_lpq.ndim != 1) and (_lpq.ndim != 2):
        raise ValueError('invalid dim for logp_xq.')
    
    if logq_xq is not None:
        _lqq = logq_xq
    elif (xq is not None) and (logq is not None):
        try:
            _lqq = logq(xq)
        except:
            raise RuntimeError('failed to get logq_xq from logq and xq.')
    else:
        raise ValueError('cannot obtain logq_xq from what you gave me.')
    _lqq = np.asarray(_lqq)
    if (_lqq.ndim != 1) and (_lqq.ndim != 2):
        raise ValueError('invalid dim for logq_xq.')
    
    if _lpq.shape != _lqq.shape:
        raise ValueError('shape of logp_xq is different from shape of logq_xq.')
    
    n_q = _lqq.size
    _lpqf = _lpq.flatten()
    _lqqf = _lqq.flatten()
    
    logr = logsumexp(_lpqf - _lqqf, b=1 / n_q)
    foo = np.exp(_lpqf - _lqqf - logr)
    logr_err = (np.var(foo) / np.mean(foo)**2 / n_q)**0.5
    
    if logr_err > 0.5:
        warnings.warn('the estimated error for logr may be unreliable since it '
                      'is larger than 0.5.')
    return logr, logr_err
