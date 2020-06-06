import numpy as np
import numbers
import random
import string
import warnings
from . import sobol
import os

cd = os.path.dirname(__file__)
df = os.path.join(cd, 'new-joe-kuo-6.21201')

__all__ = ['string', 'check_state', 'split_state', 'uniform',
           'multivariate_normal', 'SystematicResampler']


# Intended for generate dask key, but currently not used, due to some dask issue
def string(length=20, prefix='BayesFast-'):
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
    elif isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    elif isinstance(seed, np.random.RandomState):
        return seed
    else:
        raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                         ' instance' % seed)


def split_state(state, n):
    n = int(n)
    if n <= 0:
        raise ValueError('n should be a positive int instead of {}.'.format(n))
    foo = state.randint(0, 4294967296, (n, 624), np.uint32)
    return [np.random.RandomState(a) for a in foo]


def uniform(low, high, size, method='auto', skip=1, dir_file=df,
            random_state=None):
    low = np.asarray(low)
    high = np.asarray(high)
    size = np.asarray(size)
    if method == 'auto':
        try:
            return uniform(low, high, size, 'sobol', skip, dir_file)
        except:
            warnings.warn('first try with method = "sobol" failed, so next I '
                          'will try to use "pseudo".', RuntimeWarning)
        try:
            return uniform(low, high, size, 'pseudo', random_state=random_state)
        except:
            raise ValueError('you select the method as "auto", but I find '
                             'neither "sobol" nor "pseudo" works here.')
    elif method == 'sobol':
        if (low.ndim == 1 and low.shape == high.shape and size.ndim == 2 and
            low.shape[0] == size.shape[1]):
            size = size[0]
        return sobol.uniform(low, high, size, skip, dir_file)
    elif method == 'pseudo':
        if low.ndim == 1 and low.shape == high.shape and size.size == 1:
            size = (int(size), low.shape[0])
        return check_state(random_state).uniform(low, high, size)
    else:
        raise ValueError('invalid value for method.')


def multivariate_normal(mean, cov, size, method='auto', skip=1, dir_file=df,
                        random_state=None):
    if method == 'auto':
        try:
            return multivariate_normal(mean, cov, size, 'sobol', skip, dir_file)
        except:
            warnings.warn('first try with method = "sobol" failed, so next I '
                          'will try to use "pseudo".', RuntimeWarning)
        try:
            return multivariate_normal(mean, cov, size, 'pseudo', 
                                       random_state=random_state)
        except:
            raise ValueError('you select the method as "auto", but I find '
                             'neither "sobol" nor "pseudo" works here.')
    elif method == 'sobol':
        return sobol.multivariate_normal(mean, cov, size, skip, dir_file)
    elif method == 'pseudo':
        return check_state(random_state).multivariate_normal(mean, cov, size)
    else:
        raise ValueError('invalid value for method.')

