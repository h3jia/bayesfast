import numpy as np
import numbers
import random
import string
import warnings
from .sobol_seq import sobol_uniform, sobol_multivariate_normal


__all__ = ['random_str', 'check_state', 'split_state', 'random_uniform',
           'random_multivariate_normal', 'resample']


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

        
def resample(logq=None, m=None, n=None, method='systematic', nodes=[1, 100], 
             weights=None, random_state=None):
    if method == 'random':
        if m is not None:
            try:
                m = int(m)
                assert m > 0
            except:
                raise ValueError('m should be a positive int.')
        else:
            try:
                logq = np.asarray(logq)
                assert logq.ndim == 1
                m = logq.shape[0]
            except:
                raise ValueError('failded to get m from logq.')
        return check_state(random_state).choice(m, n)
    elif method == 'systematic':
        try:
            logq = np.asarray(logq)
            assert logq.ndim == 1
        except:
            raise ValueError('logq should be a 1-d array.')
        try:
            n = int(n)
            assert n > 0
        except:
            raise ValueError('n should be a positive int.')
        if nodes is None and weights is None:
            nodes = np.array([0, 100])
            weights = np.array([1])
        elif nodes is not None and weights is None:
            try:
                nodes = np.asarray(nodes)
                assert nodes.shape == (2,)
                assert nodes[0] >= 0 and nodes[1] <= 100
            except:
                raise ValueError('invalid value for nodes and/or weights.')
            weights = np.array([1])
        else:
            try:
                nodes = np.asarray(nodes)
                weights = np.asarray(weights)
                assert nodes.ndim == 1 and weights.ndim == 1
                assert nodes.shape[0] == weights.shape[0] + 1
                assert np.all(weights > 0)
                assert np.all(np.diff(nodes) > 0)
                assert nodes[0] >= 0 and nodes[1] <= 100
            except:
                raise ValueError('invalid value for nodes and/or weights.')
            weights /= np.sum(weights)
        n_w = (n * weights).astype(np.int)
        n_w[-1] += n - np.sum(n_w)
        n_c = np.cumsum(np.insert(n_w, 0, 0))
        result = np.empty(n, dtype=np.int)
        a = len(n_w)
        q = len(logq)
        for i in range(a):
            ep = (i == (a - 1))
            foo = np.linspace(nodes[i] * (q - 1) / 100, nodes[i + 1] * (q - 1) / 
                              100, n_w[i], ep)
            result[n_c[i]:n_c[i + 1]] = foo.astype(np.int)
        if np.unique(result).size < result.size:
            warnings.warn(
                '{:.1f}% of the resampled points are not unique. Please '
                'consider giving me more points.'.format(100 - 
                np.unique(result).size / result.size * 100), RuntimeWarning)
        return np.argsort(logq)[result]
    else:
        raise ValueError(
            'Sorry I do not know how to do {} resample.'.format(method))
    