import numpy as np
from ..utils.vectorize import vectorize
from scipy.special import logsumexp
from scipy.optimize import root_scalar
from scipy.stats import multivariate_normal
from ..utils.acor import integrated_time
from ..utils.random import check_state
import warnings

__all__ = ['BS', 'WarpBS']


#TODO: imporve shape checks
#TODO: enable reusing logp_xp in WarpBS


def BS(xp=None, xq=None, logp=None, logq=None, adapt_n_q=False, 
       logp_xp=None, logp_xq=None, logq_xp=None, logq_xq=None, 
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
    
    if _lpp.shape != _lqp.shape:
        raise ValueError('shape of logp_xp is different from shape of logq_xp.')
    if _lpq.shape != _lqq.shape:
        raise ValueError('shape of logp_xq is different from shape of logq_xq.')
    
    n_p = _lpp.size
    n_q = _lqq.size
    _lppf = _lpp.flatten()
    _lpqf = _lpq.flatten()
    _lqpf = _lqp.flatten()
    _lqqf = _lqq.flatten()
    
    _alpha = _lqpf - _lppf - np.log(n_p / n_q)
    _beta = _lpqf - _lqqf + np.log(n_p / n_q)
    
    def score(logr):
        _a = logsumexp(logr + _alpha - logsumexp(np.array((logr + _alpha, 
                       np.zeros_like(_alpha))), axis=0))
        _b = logsumexp(-logr + _beta - logsumexp(np.array((-logr + _beta, 
                       np.zeros_like(_beta))), axis=0))
        return _a - _b
    logr = root_scalar(score, x0=0., x1=5.).root
    
    #print('COMPARE', logr, score(logr), score(33.05))
    
    _f1 = np.exp(_lpqf - logr - logsumexp(np.array((_lpqf - logr + 
                 np.log(n_p / (n_p + n_q)), _lqqf + np.log(n_q / (n_p + n_q)))), 
                 axis=0))
    _f2 = np.exp(_lqpf - logsumexp(np.array((_lppf - logr + 
                 np.log(n_p / (n_p + n_q)), _lqpf + np.log(n_q / (n_p + n_q)))), 
                 axis=0))
    tau = integrated_time(_f2.reshape(_lpp.shape)[..., np.newaxis])[0]
    re2_p = tau * np.var(_f2) / np.mean(_f2)**2 / n_p
    re2_q = np.var(_f1) / np.mean(_f1)**2 / n_q
    logr_err = (re2_p + re2_q)**0.5
    #print(tau * np.var(_f2) / np.mean(_f2)**2 / n_p, np.var(_f1) / np.mean(_f1)**2 / n_q)
    
    if logr_err > 0.5:
        warnings.warn('the estimated error for logr may be unreliable since it '
                      'is larger than 0.5.')
    if adapt_n_q:
        return logr, logr_err, re2_p, re2_q
    else:
        return logr, logr_err


def adapt_n_q(re2_p, re2_q, n_q0, n_call_p, target_q_error=0.1, max_q_call=0.1):
    try:
        target_q_error = float(target_q_error)
        max_q_call = float(max_q_call)
        if target_q_error == 0.:
            target_q_error = 1e-16
        if max_q_call == 1.:
            max_q_call = 1 - 1e-16
        assert (0. < target_q_error < 1.) and (0. < max_q_call < 1.)
    except:
        raise ValueError('invalid values for target_q_error and/or max_q_call.')
    from_error = n_q0 * (1 - target_q_error) * re2_q / target_q_error / re2_p
    from_call = n_call_p * max_q_call / (1 - max_q_call)
    return int(max(n_q0, min(from_error, from_call)))
    

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
