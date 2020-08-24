import numpy as np
from scipy.special import logsumexp
from scipy.optimize import root_scalar
from ..utils import integrated_time
import warnings

__all__ = ['bridge']


def bridge(logp_p, logp_q, logq_p, logq_q):
    try:
        lpp = np.asarray(logp_p)
        lpq = np.asarray(logp_q)
        lqp = np.asarray(logq_p)
        lqq = np.asarray(logq_q)
    except Exception:
        raise ValueError('invalid value for the inputs.')
    
    if (lqq.ndim != 1) and (lqq.ndim != 2):
        raise ValueError(
            'dim of logq_q should be 1 or 2, instead of {}.'.format(lqq.ndim))
    if (lpp.ndim != 1) and (lpp.ndim != 2):
        raise ValueError(
            'dim of logp_p should be 1 or 2, instead of {}.'.format(lpp.ndim))
    if lpp.shape != lqp.shape:
        raise ValueError('shape of logp_p, {}, is different from shape of '
                         'logq_p, {}.'.format(lpp.shape, lqp.shape))
    if lpq.shape != lqq.shape:
        raise ValueError('shape of logp_q, {}, is different from shape of '
                         'logq_q, {}.'.format(lpq.shape, lqq.shape))
    
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
    f1 = np.exp(lpqf - logr - logsumexp(np.array((lpqf - logr +
                 np.log(n_p / (n_p + n_q)), lqqf + np.log(n_q / (n_p + n_q)))),
                 axis=0))
    f2 = np.exp(lqpf - logsumexp(np.array((lppf - logr +
                 np.log(n_p / (n_p + n_q)), lqpf + np.log(n_q / (n_p + n_q)))),
                 axis=0))
    re2_q = np.var(f1) / np.mean(f1)**2 / n_q
    
    tau_uf = integrated_time(f2.reshape(lpp.shape)[..., np.newaxis])[0]
    re2_p_uf = tau_uf * np.var(f2) / np.mean(f2)**2 / n_p
    logr_err_uf = (re2_p_uf + re2_q)**0.5
    
    tau_f = integrated_time(f2[..., np.newaxis])[0]
    re2_p_f = tau_f * np.var(f2) / np.mean(f2)**2 / n_p
    logr_err_f = (re2_p_f + re2_q)**0.5
    
    diff_err = abs(logr_err_f - logr_err_uf) / min(logr_err_f, logr_err_uf)
    logr_err = max(logr_err_f, logr_err_uf)
    
    if diff_err > 0.25:
        warnings.warn('the estimated error for logr may be unreliable, '
                      'since flattening before estimating tau makes the '
                      'result differ by more than 25%.', RuntimeWarning)
    if logr_err > 0.25:
        warnings.warn('the estimated error for logr may be unreliable, '
                      'since the result is larger than 0.25.', RuntimeWarning)
    return logr, logr_err
