import numpy as np
from scipy.special import logsumexp
from ..utils import integrated_time
import warnings

__all__ = ['harmonic']


def harmonic(logp_p, logq_p):
    try:
        lpp = np.asarray(logp_p)
        lqp = np.asarray(logq_p)
    except Exception:
        raise ValueError('invalid value for the inputs.')

    if (lpp.ndim != 1) and (lpp.ndim != 2):
        raise ValueError(
            'dim of logp_p should be 1 or 2, instead of {}.'.format(lpp.ndim))
    if lpp.shape != lqp.shape:
        raise ValueError('shape of logp_p, {}, is different from shape of '
                         'logq_p, {}.'.format(lpp.shape, lqp.shape))

    n_p = lpp.size
    lppf = lpp.flatten()
    lqpf = lqp.flatten()

    logr = -logsumexp(lqpf - lppf, b=1 / n_p)
    foo = np.exp(lqpf - lppf + logr)

    tau_uf = integrated_time(foo.reshape(lpp.shape)[..., np.newaxis])[0]
    logr_err_uf = (tau_uf * np.var(foo) / np.mean(foo)**2 / n_p)**0.5

    tau_f = integrated_time(foo[..., np.newaxis])[0]
    logr_err_f = (tau_f * np.var(foo) / np.mean(foo)**2 / n_p)**0.5

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
