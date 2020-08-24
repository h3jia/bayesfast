#-------------------------------------------------------------------------------
#
#  Adapted from Scipy's interpolate.py and _cubic.py
#
#  URL: https://github.com/scipy/scipy/blob/master/scipy/interpolate/interpolate.py
#       https://github.com/scipy/scipy/blob/master/scipy/interpolate/_cubic.py
#
#-------------------------------------------------------------------------------

import numpy as np
from . import _cubic
from scipy.stats import linregress
from scipy.linalg import solve_banded
import warnings

__all__ = ['cubic_spline']


class cubic_spline:
    """Cubic spline data interpolator.
    Interpolate data with a piecewise cubic polynomial which is twice
    continuously differentiable [1]_. The result is represented as a `PPoly`
    instance with breakpoints matching the given data.
    Parameters
    ----------
    x : array_like, shape (n,)
        1-d array containing values of the independent variable.
        Values must be real, finite and in strictly increasing order.
    y : array_like
        Array containing values of the dependent variable. It can have
        arbitrary number of dimensions, but the length along ``axis``
        (see below) must match the length of ``x``. Values must be finite.
    Attributes
    ----------
    x : ndarray, shape (n,)
        Breakpoints. The same ``x`` which was passed to the constructor.
    c : ndarray, shape (4, n-1, ...)
        Coefficients of the polynomials on each segment. The trailing
        dimensions match the dimensions of `y`, excluding ``axis``.
        For example, if `y` is 1-d, then ``c[k, i]`` is a coefficient for
        ``(x-x[i])**(3-k)`` on the segment between ``x[i]`` and ``x[i+1]``.
    axis : int
        Interpolation axis. The same axis which was passed to the
        constructor.
    Methods
    -------
    __call__
    derivative
    antiderivative
    integrate
    roots
    References
    ----------
    .. [1] `Cubic Spline Interpolation
            <https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation>`_
            on Wikiversity.
    .. [2] Carl de Boor, "A Practical Guide to Splines", Springer-Verlag, 1978.
    """
    __slots__ = ('_x', '_n', '_fun', '_c', '_y', '_mask')
    
    def __init__(self, x_all, fun, bins=100, edge_bins=1, edge_points=10,
                 max_width=5, split=4, max_add=5, save_fun=False):
        # TODO: add all kinds of checks
        x_all = np.ascontiguousarray(x_all)
        edge_bins = np.min((edge_bins, bins // 4))
        # mask = np.empty((0, 4))
        self._x = np.unique(np.percentile(
            x_all, np.linspace(0, 100, bins + 1)[edge_bins:-edge_bins]))
        self._y = fun(self._x)
        # mask = self._regularize_y()
        self._n = self._x.shape[0]
        
        x_edge_1 = np.percentile(x_all[x_all < self._x[edge_bins]] - self._x[0],
                                 np.linspace(0, 100, edge_points + 2)[1:-1])
        y_edge_1 = fun(x_edge_1 + self._x[0]) - self._y[0]
        k_edge_1 = np.sum(x_edge_1 * y_edge_1) / np.sum(x_edge_1 * x_edge_1)
        x_edge_2 = np.percentile(
            x_all[x_all > self._x[-edge_bins - 1]] - self._x[-1],
            np.linspace(0, 100, edge_points + 2)[1:-1])
        y_edge_2 = fun(x_edge_2 + self._x[-1]) - self._y[-1]
        k_edge_2 = np.sum(x_edge_2 * y_edge_2) / np.sum(x_edge_2 * x_edge_2)
        
        diff = np.diff(self._x)
        diff_r = diff / np.mean(diff) # index from 0 to self._n - 2
        
        i_1 = 0
        while diff_r[i_1] > max_width:
            i_1 += 1
            if i_1 >= self._n - 2:
                break
        
        i_2 = self._n - 2
        while diff_r[i_2] > max_width:
            i_2 -= 1
            if i_2 <= 0:
                break
                
        if i_1 <= i_2:
            sparse_index = np.where(diff_r[i_1:(i_2 + 1)] > max_width)[0] + i_1
            if sparse_index.size:
                x_aug = np.empty(0)
                for j in sparse_index:
                    n_j = int(np.ceil(diff_r[j] / split))
                    x_aug_j = np.linspace(
                        self._x[j], self._x[j + 1], n_j + 1)[1:-1]
                    x_aug = np.concatenate((x_aug, x_aug_j))
                insert_index = np.searchsorted(self._x, x_aug)
                self._x = np.insert(self._x, insert_index, x_aug)
                self._y = np.insert(self._y, insert_index, fun(x_aug))
                # self._regularize_y()
                self._n = self._x.shape[0]
        else:
            raise ValueError
            
        self._fit(edge_bins, k_edge_1, k_edge_2)
        check = self._check()
        
        add_points = 0
        while (not np.all(check)) and add_points < max_add:
            x_aug = np.empty(0)
            for j in np.where(check == 0)[0]:
                x_aug_j = np.linspace(
                    self._x[j], self._x[j + 1], split + 1)[1:-1]
                x_aug = np.concatenate((x_aug, x_aug_j))
            insert_index = np.searchsorted(self._x, x_aug)
            self._x = np.insert(self._x, insert_index, x_aug)
            self._y = np.insert(self._y, insert_index, fun(x_aug))
            if add_points == max_add - 1:
                self._regularize_y()
            self._n = self._x.shape[0]
            self._fit(edge_bins, k_edge_1, k_edge_2)
            check = self._check()
            add_points += 1
            
        if not np.all(check):
            bad_index = np.where(check == 0)[0] + 1
            for i_b in bad_index:
                self._c[i_b, 0] = 0
                self._c[i_b, 1] = 0
                self._c[i_b, 2] = (self._y[i_b] - self._y[i_b - 1]) / (
                    self._x[i_b] - self._x[i_b - 1])
                self._c[i_b, 3] = self._y[i_b - 1]
            check = self._check()
            
        if not np.all(check):
            warnings.warn(RuntimeWarning("Not all the intervals are monotone."))
        
        if save_fun:
            self._fun = fun
            
        # TODO: check the edge intervals
    
    def _fit(self, edge_bins, k_edge_1, k_edge_2):
        self._c = np.zeros((self._n + 1, 4))
        self._c[0, 2:] = (k_edge_1, self._y[0])
        self._c[-1, 2:] = (k_edge_2, self._y[-1])
        
        dx = np.diff(self._x)
        slope = np.diff(self._y) / dx

        # Find derivative values at each x[i] by solving a tridiagonal
        # system.
        A = np.zeros((3, self._n))  # This is a banded matrix representation.
        b = np.empty(self._n)

        # Filling the system for i = 1...n-2
        #                         (x[i-1] - x[i]) * s[i-1] +\
        # 2 * ((x[i] - x[i-1]) + (x[i+1] - x[i])) * s[i]   +\
        #                         (x[i] - x[i-1]) * s[i+1] =\
        # 3 * ((x[i+1] - x[i])*(y[i] - y[i-1])/(x[i] - x[i-1]) +\
        #      (x[i] - x[i-1])*(y[i+1] - y[i])/(x[i+1] - x[i]))

        A[1, 1:-1] = 2 * (dx[:-1] + dx[1:])  # The diagonal
        A[0, 2:] = dx[:-1]                   # The upper diagonal
        A[-1, :-2] = dx[1:]                  # The lower diagonal

        b[1:-1] = 3 * (dx[1:] * slope[:-1] + dx[:-1] * slope[1:])

        A[1, 0] = 1
        A[0, 1] = 0
        b[0] = k_edge_1

        A[1, -1] = 1
        A[-1, -2] = 0
        b[-1] = k_edge_2

        s = solve_banded((1, 1), A, b, overwrite_ab=True,
                         overwrite_b=True, check_finite=False)

        t = (s[:-1] + s[1:] - 2 * slope) / dx
        self._c[1:-1, 0] = t / dx
        self._c[1:-1, 1] = (slope - s[:-1]) / dx - t
        self._c[1:-1, 2] = s[:-1]
        self._c[1:-1, 3] = self._y[:-1]
        
    def _check(self):
        out = np.empty(self._n - 1, dtype=np.uint8)
        _cubic.is_monotone(self._c, self._x, out)
        return out
    
    def _regularize_y(self):
        x_diff = np.diff(self._x)
        y_diff = np.diff(self._y)
        k = y_diff / x_diff
        bad_index = np.where(k < 1e-10)[0]
        n_b = bad_index.size
        
        while n_b > 0:
            while n_b > 0:
                i_b = 0
                start_b = np.max(bad_index[i_b] - 1, 0)
                while i_b < n_b - 1:
                    if bad_index[i_b + 1] - bad_index[i_b] <= 2:
                        i_b += 1
                    else:
                        break
                end_b = np.min((bad_index[i_b] + 1, k.size - 1))
                k_b = (self._y[end_b + 1] - self._y[start_b]) / (
                    self._x[end_b + 1] - self._x[start_b])
                for j_b in range(start_b + 1, end_b + 1):
                    self._y[j_b] = self._y[start_b] + k_b * (
                        self._x[j_b] - self._x[start_b])
                bad_index = bad_index[(i_b + 1):]
                n_b = bad_index.size
            y_diff = np.diff(self._y)
            k = y_diff / x_diff
            bad_index = np.where(k < 1e-8)[0]
            n_b = bad_index.size
    
    def evaluate(self, x):
        x = np.ascontiguousarray(x)
        out = np.empty_like(x)
        _cubic.evaluate(self._c, self._x, x, out)
        return out
        
    __call__ = evaluate
    
    def derivative(self, x):
        x = np.ascontiguousarray(x)
        out = np.empty_like(x)
        _cubic.derivative(self._c, self._x, x, out)
        return out
    
    def solve(self, y):
        y = np.ascontiguousarray(y)
        out = np.empty_like(y)
        _cubic.solve(self._c, self._x, self._y, y, out)
        return out
