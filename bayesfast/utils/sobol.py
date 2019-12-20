from ._sobol import _sobol
import numpy as np
from scipy.stats import norm
import os

__all__ = ['uniform', 'multivariate_normal']

cd = os.path.dirname(__file__)
df = os.path.join(cd, 'new-joe-kuo-6.21201')


def uniform(low, high, size, skip=1, dir_file=df):
    low = np.atleast_1d(low)
    high = np.atleast_1d(high)
    if low.ndim == 1 and low.shape == high.shape:
        d = low.shape[0]
    else:
        raise ValueError('low and high should be 1-d arraies with the same '
                         'shape, but you give me low.shape = {}, high.shape = '
                         '{}.'.format(low.shape, high.shape))
    if not os.path.isfile(dir_file):
        raise ValueError('cannot find the direction data file at '
                         'dir_file="{}"'.format(dir_file))
    d_max = len(open(df, 'r').readlines())
    if d > d_max:
        raise NotImplementedError(
            'd = {} is not supported yet, as we have d_max = {}, so please '
            'use pseudo random numbers for now.'.format(d, d_max))
    try:
        size = int(size)
        assert size > 0
    except:
        raise ValueError(
            'size should be a positive int, instead of {}.'.format(size))
    try:
        skip = int(skip)
        assert skip >= 0
    except:
        raise ValueError('skip should be a non-negative int, instead of '
                         '{}.'.format(skip))
    n_all = size + skip
    points = np.empty((n_all, d))
    _sobol(n_all, d, bytes(dir_file, 'utf8'), points)
    points = points[skip:]
    points = low + (high - low) * points
    return points
    

def multivariate_normal(mean, cov, size, skip=1, dir_file=df):
    mean = np.atleast_1d(mean)
    cov = np.atleast_2d(cov)
    d = mean.shape[0]
    if not (mean.shape == (d,) and cov.shape == (d, d)):
        raise ValueError(
            'the shape of mean is not consistent with the shape of cov.')
    points = uniform(np.zeros(d), np.ones(d), size, skip, dir_file)
    points = norm.ppf(points)
    a, w = np.linalg.eigh(cov)
    points = mean + (points * a**0.5) @ w.T
    return points
