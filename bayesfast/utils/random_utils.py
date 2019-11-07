import numpy as np
import numbers
import random
import string
from .sobol_seq import sobol_uniform, sobol_multivariate_normal


__all__ = ['random_str', 'check_state', 'split_state', 'random_uniform',
           'random_multivariate_normal']


def random_str(length=20, prefix='BayesFast:'):
    return (str(prefix) + ''.join(random.SystemRandom().choice(
        string.ascii_letters + string.digits) for _ in range(length)))


def check_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def split_state(state, n):
    n = int(n)
    if n <= 0:
        raise ValueError('n should be a positive int instead of {}.'.format(n))
    foo = state.randint(0, 4294967296, (n, 624), np.uint32)
    return [np.random.RandomState(a) for a in foo]


def random_uniform(low, high, size, method='auto', seed=None):
    if size is None:
        method = 'pseudo'
    if method == 'auto':
        try:
            size = np.atleast_1d(size).astype(np.int)
        except:
            raise ValueError('invalid value for size.')
        if size.ndim <= 2 and 1 <= size.shape[-1] <= 40:
            method = 'sobol'
            if seed is None:
                seed = 1
            try:
                seed = int(seed)
                assert seed > 0
            except:
                raise ValueError('invalid value for seed.')
        else:
            method = 'pseudo'
    if method == 'sobol':
        return sobol_uniform(low, high, size, seed)
    elif method == 'pseudo':
        return check_state(seed).uniform(low, high, size)
    else:
        raise ValueError('invalid value for method.')


def random_multivariate_normal(mean, cov, size, method='auto', seed=None):
    if size is None:
        method = 'pseudo'
    if method == 'auto':
        try:
            size = np.atleast_1d(size).astype(np.int)
        except:
            raise ValueError('invalid value for size.')
        d = mean.shape[-1]
        if (mean.shape == (d,) and cov.shape == (d, d) and size.ndim == 1 and 
            1 <= d <= 40):
            method = 'sobol'
            if seed is None:
                seed = 1
            try:
                seed = int(seed)
                assert seed > 0
            except:
                raise ValueError('invalid value for seed.')
        else:
            method = 'pseudo'
    if method == 'sobol':
        return sobol_multivariate_normal(mean, cov, size, seed)
    elif method == 'pseudo':
        return check_state(seed).multivariate_normal(mean, cov, size)
    else:
        raise ValueError('invalid value for method.')


def stratified_resample_1(logq, n, a=1, b=100):
    logq = np.asarray(logq).flatten()
    m = logq.size
    i_finite = np.arange(m)[np.isfinite(logq)]
    logq = logq[np.isfinite(logq)].copy()
    mf = logq.size
    
    _a = int(mf * a / 100 - 1)
    _b = int(mf * b / 100 - 1)
    
    i = np.linspace(_a, _b, n).astype(np.int)
    return i_finite[np.argsort(logq)[i]]


def stratified_resample_2(logq, n, a=1, b=20, c=100, f=0.8):
    logq = np.asarray(logq).flatten()
    m = logq.size
    i_finite = np.arange(m)[np.isfinite(logq)]
    logq = logq[np.isfinite(logq)].copy()
    mf = logq.size
    
    _a = int(mf * a / 100 - 1)
    _b = int(mf * b / 100 - 1)
    _c = int(mf * c / 100 - 1)
    
    n_1 = int(f * n)
    n_2 = n - n_1
    i = np.concatenate((np.linspace(_a, _b, n_1), 
                        np.linspace(_b, _c, n_2 + 1)[1:])).astype(np.int)
    return i_finite[np.argsort(logq)[i]]
        