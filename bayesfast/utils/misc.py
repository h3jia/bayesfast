import numpy as np
import warnings

__all__ = ['all_isinstance', 'make_positive', 'SystematicResampler']


def all_isinstance(iterable, class_or_tuple):
    return (hasattr(iterable, '__iter__') and 
            all(isinstance(i, class_or_tuple) for i in iterable))


def make_positive(A, max_cond=1e5):
    a, w = np.linalg.eigh(A) # a: all the eigenvalues, in ascending order
    if a[-1] <= 0:
        raise ValueError('all the eigenvalues are non-positive.')
    i = np.argmax(a > a[-1] / max_cond)
    a[:i] = a[i]
    return w @ np.diag(a) @ w.T


class SystematicResampler:
    
    def __init__(self, nodes=[1, 100], weights=None, require_unique=True):
        try:
            self._nodes = np.asarray(nodes, dtype=np.float)
            assert self._nodes.ndim == 1 and self._nodes.size > 1
            assert np.all(np.diff(self._nodes) > 0)
            assert self._nodes[0] >= 0 and self._nodes[-1] <= 100
            self._n_node = self._nodes.size
        except Exception:
            raise ValueError('invalid value for nodes.')
        
        if weights is None:
            self._weights = np.ones(self._n_node - 1) / (self._n_node - 1)
        else:
            try:
                self._weights = np.asarray(weights, dtype=np.float)
                assert np.all(self._weights) > 0
                assert self._weights.ndim == 1
                assert self._weights.size == self._n_node - 1
                self._weights = self._weights / np.sum(self._weights)
            except Exception:
                raise ValueError('invalid value for weights.')
        
        self._require_unique = bool(require_unique)
    
    def run(self, a, n):
        try:
            a = np.asarray(a, dtype=np.float)
            assert a.ndim == 1
        except Exception:
            raise ValueError('invalid value for a.')
        try:
            n = int(n)
            assert n > 0
        except Exception:
            raise ValueError('invalid value for n.')
        
        n_w = (n * self._weights).astype(np.int)
        n_w[-1] += n - np.sum(n_w)
        n_c = np.cumsum(np.insert(n_w, 0, 0))
        i_all = np.empty(n, dtype=np.int)
        m = len(a)
        
        for j in range(self._n_node - 1):
            ep = (j == self._n_node - 2)
            i_j = np.linspace(
                self._nodes[j] * (m - 1) / 100, self._nodes[j + 1] * (m - 1) /
                100, n_w[j], ep)
            i_all[n_c[j]:n_c[j + 1]] = i_j.astype(np.int)
        if np.unique(i_all).size < i_all.size:
            message = ('{:.1f}% of the resampled points are not unique. Please '
                       'consider giving me more points.'.format(100 - 
                       np.unique(i_all).size / i_all.size * 100))
            if self._require_unique:
                raise RuntimeError(message)
            else:
                warnings.warn(message, RuntimeWarning)
        return np.argsort(a)[i_all]
    
    __call__ = run
