import numpy as np
from ..utils.vectorize import vectorize
from scipy.special import logsumexp
from scipy.optimize import root_scalar
from scipy.stats import multivariate_normal
from ..utils import integrated_time
from ..utils.random import check_state
import warnings

#__all__ = ['BS', 'WarpBS']
__all__ = ['BS']


#TODO: enable reusing logp_xp in WarpBS


def BS(xp=None, xq=None, logp=None, logq=None, adapt_n_q=False, 
       logp_xp=None, logp_xq=None, logq_xp=None, logq_xq=None, 
       logp_vectorized=True, logq_vectorized=True):
    _logp = (logp if logp_vectorized else 
             lambda x: np.apply_along_axis(logp, -1, x))
    _logq = (logq if logq_vectorized else 
             lambda x: np.apply_along_axis(logq, -1, x))
    
    if logp_xp is not None:
        _lpp = np.asarray(logp_xp)
    elif (xp is not None) and callable(logp):
        xp = np.asarray(xp)
        try:
            _lpp = np.asarray(_logp(xp))
        except:
            raise RuntimeError('failed to get logp_xp from logp and xp.')
        if xp.shape[:-1] == _lpp.shape:
            pass
        elif (*xp.shape[:-1], 1) == _lpp.shape:
            _lpp = _lpp[..., 0]
        else:
            raise ValueError(
                'the shape of xp, {}, is not consistent with the shape of '
                'logp_xp from logp and xp, {}.'.format(xp.shape, _lpp.shape))
    else:
        raise ValueError('cannot obtain logp_xp from what you gave me.')
    if (_lpp.ndim != 1) and (_lpp.ndim != 2):
        raise ValueError(
            'dim of logp_xp should be 1 or 2, instead of {}.'.format(_lpp.ndim))
    
    if logp_xq is not None:
        _lpq = np.asarray(logp_xq)
    elif (xq is not None) and callable(logp):
        xq = np.asarray(xq)
        try:
            _lpq = np.asarray(_logp(xq))
        except:
            raise RuntimeError('failed to get logp_xq from logp and xq.')
        if xq.shape[:-1] == _lpq.shape:
            pass
        elif (*xq.shape[:-1], 1) == _lpq.shape:
            _lpq = _lpq[..., 0]
        else:
            raise ValueError(
                'the shape of xq, {}, is not consistent with the shape of '
                'logp_xq from logp and xq, {}.'.format(xq.shape, _lpq.shape))
    else:
        raise ValueError('cannot obtain logp_xq from what you gave me.')
    if (_lpq.ndim != 1) and (_lpq.ndim != 2):
        raise ValueError(
            'dim of logp_xq should be 1 or 2, instead of {}.'.format(_lpq.ndim))
    
    if logq_xp is not None:
        _lqp = np.asarray(logq_xp)
    elif (xp is not None) and callable(logq):
        xp = np.asarray(xp)
        try:
            _lqp = np.asarray(_logq(xp))
        except:
            raise RuntimeError('failed to get logq_xp from logq and xp.')
        if xp.shape[:-1] == _lqp.shape:
            pass
        elif (*xp.shape[:-1], 1) == _lqp.shape:
            _lqp = _lqp[..., 0]
        else:
            raise ValueError(
                'the shape of xp, {}, is not consistent with the shape of '
                'logq_xp from logq and xp, {}.'.format(xp.shape, _lqp.shape))
    else:
        raise ValueError('cannot obtain logq_xp from what you gave me.')
    if (_lqp.ndim != 1) and (_lqp.ndim != 2):
        raise ValueError(
            'dim of logq_xp should be 1 or 2, instead of {}.'.format(_lqp.ndim))
    
    if logq_xq is not None:
        _lqq = np.asarray(logq_xq)
    elif (xq is not None) and callable(logq):
        xq = np.asarray(xq)
        try:
            _lqq = np.asarray(_logq(xq))
        except:
            raise RuntimeError('failed to get logq_xq from logq and xq.')
        if xq.shape[:-1] == _lqq.shape:
            pass
        elif (*xq.shape[:-1], 1) == _lqq.shape:
            _lqq = _lqq[..., 0]
        else:
            raise ValueError(
                'the shape of xq, {}, is not consistent with the shape of '
                'logq_xq from logq and xq, {}.'.format(xq.shape, _lqq.shape))
    else:
        raise ValueError('cannot obtain logq_xq from what you gave me.')
    if (_lqq.ndim != 1) and (_lqq.ndim != 2):
        raise ValueError(
            'dim of logq_xq should be 1 or 2, instead of {}.'.format(_lqq.ndim))
    
    if _lpp.shape != _lqp.shape:
        raise ValueError('shape of logp_xp, {}, is different from shape of '
                         'logq_xp, {}.'.format(_lpp.shape, _lqp.shape))
    if _lpq.shape != _lqq.shape:
        raise ValueError('shape of logp_xq, {}, is different from shape of '
                         'logq_xq, {}.'.format(_lpq.shape, _lqq.shape))
    
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


"""
NOTE: below is some legacy code that is not compatible with current BayesFast
      we will revise it later
"""

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
