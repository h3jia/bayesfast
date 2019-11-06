import numpy as np
from ..utils.vectorize import vectorize
from scipy.special import logsumexp
from ..utils.acor import integrated_time
import warnings

__all__ = ['HM']


def HM(xp=None, logp=None, logq=None, logp_xp=None, logq_xp=None, 
       logp_vectorized=True, logq_vectorized=True, logp_vectorize_level=1, 
       logq_vectorize_level=1):
    if not logp_vectorized:
        logp = vectorize(logp, logp_vectorize_level)
    if not logq_vectorized:
        logq = vectorize(logq, logq_vectorize_level)
    
    if logp_xp is not None:
        _lpp = logp_xp
    elif (xp is not None) and (logp is not None):
        try:
            _lpp = logp(xp)
        except:
            raise RuntimeError('failed to get logp_xp from logp and xp.')
    else:
        raise ValueError('cannot obtain logp_xp from what you gave me.')
    _lpp = np.asarray(_lpp)
    if (_lpp.ndim != 1) and (_lpp.ndim != 2):
        raise ValueError('invalid dim for logp_xp.')
    
    if logq_xp is not None:
        _lqp = logq_xp
    elif (xp is not None) and (logq is not None):
        try:
            _lqp = logq(xp)
        except:
            raise RuntimeError('failed to get logq_xp from logq and xp.')
    else:
        raise ValueError('cannot obtain logq_xp from what you gave me.')
    _lqp = np.asarray(_lqp)
    if (_lqp.ndim != 1) and (_lqp.ndim != 2):
        raise ValueError('invalid dim for logq_xp.')
    
    if _lpp.shape != _lqp.shape:
        raise ValueError('shape of logp_xp is different from shape of logq_xp.')
    
    n_p = _lpp.size
    _lppf = _lpp.flatten()
    _lqpf = _lqp.flatten()
    
    logr = -logsumexp(_lqpf - _lppf, b=1 / n_p)
    foo = np.exp(_lqpf - _lppf + logr)
    tau = integrated_time(foo.reshape(_lpp.shape)[..., np.newaxis])[0]
    logr_err = (tau * np.var(foo) / np.mean(foo)**2 / n_p)**0.5
    
    if logr_err > 0.5:
        warnings.warn('the estimated error for logr may be unreliable since it '
                      'is larger than 0.5.')
    return logr, logr_err
