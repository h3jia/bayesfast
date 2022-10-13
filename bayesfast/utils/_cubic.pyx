#-------------------------------------------------------------------------------
#
#  Adapted from Scipy's _ppoly.pyx.
#
#  URL: https://github.com/scipy/scipy/blob/master/scipy/interpolate/_ppoly.pyx
#
#-------------------------------------------------------------------------------

"""
Routines for evaluating and manipulating piecewise polynomials in local power basis.
"""

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

ctypedef np.uint8_t uint8  # we use this for boolean type
cdef extern from "numpy/npy_math.h":
    double nan "NPY_NAN"


#------------------------------------------------------------------------------
# Piecewise power basis polynomials
#------------------------------------------------------------------------------

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef int find_interval(const double* x, int m, double xval, int prev_interval=-1) nogil:
    """
    Find an interval such that x[interval - 1] <= xval < x[interval].

    Assumeing that x is sorted in the ascending order. If xval < x[0], then interval = 0, if xval >
    x[-1] then interval = m.

    Parameters
    ----------
    x : ndarray of double, shape (m,)
        Piecewise polynomial breakpoints sorted in ascending order.
    m : int
        Shape of x.
    xval : double
        Point to find.
    prev_interval : int, optional
        Interval where a previous point was found.

    Returns
    -------
    interval : int
        Suitable interval or -1 if nan.
    """
    cdef int high, low, mid, interval
    cdef double a, b

    a = x[0]
    b = x[m - 1]

    interval = prev_interval
    if interval < 0 or interval > m:
        interval = m // 2

    if not (a <= xval < b):
        if xval < a:
            # below
            interval = 0
        elif xval >= b:
            # above
            interval = m
        else:
            # nan
            interval = -1
    else:
        # Find the interval the coordinate is in (binary search with locality)
        if xval >= x[interval - 1]:
            low = interval
            high = m - 1
        else:
            low = 1
            high = interval - 1

        if xval < x[low]:
            high = low

        while low < high:
            mid = (high + low) // 2
            if xval < x[mid]:
                # mid < high
                high = mid
            elif xval >= x[mid + 1]:
                low = mid + 2
            else:
                # x[mid] <= xval < x[mid+1]
                low = mid + 1
                break

        interval = low

    return interval


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline double _evaluate(const double* c, double x) nogil:
    return c[0] * x * x * x + c[1] * x * x + c[2] * x + c[3]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline double _derivative(const double* c, double x) nogil:
    return 3 * c[0] * x * x + 2 * c[1] * x + c[2]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double solve_newton(const double* c, double yp, double x0, double x1, double tol=1e-10) nogil:
    cdef int i ##### TODO #####
    cdef double x, y
    i = 0
    x = (x1 - x0) / 2
    y = _evaluate(c, x) - yp
    while not (y < tol and y > -tol):
        x = x - y / _derivative(c, x)
        y = _evaluate(c, x) - yp
        i += 1
        if i >= 100:
            x = nan
            break
    if not (x >= 0 and x <= x1- x0):
        x = nan ##### TODO #####
    return x


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double solve_bisect(const double* c, double yp, double x0, double x1, double tol=1e-10) nogil:
    cdef int i ##### TODO #####
    cdef double a, b, x
    i = 0
    a = 0.
    b = x1 - x0
    x = (a + b) / 2
    y = _evaluate(c, x) - yp
    while not (y < tol and y > -tol):
        if y > 0:
            b = x
            x = (a + b) / 2
            y = _evaluate(c, x) - yp
        else:
            a = x
            x = (a + b) / 2
            y = _evaluate(c, x) - yp
        i += 1
        if i >= 100:
            x = nan
            break
    return x


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef uint8 _is_monotone(const double* c, double x0, double x1) nogil:
    A = _derivative(c, x0)
    B = _derivative(c, x1)
    C = 3 * c[0] * x0 + c[1]
    D = 3 * c[0] * x1 + c[1]
    delta = c[1] * c[1] - 3 * c[0] * c[2]
    if A > 0 and B > 0 and (C * D) >= 0:
        return 1
    elif c[0] > 0 and delta < 0:
        return 1
    else:
        return 0


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def evaluate(const double[:,::1] c, const double[::1] x, const double[::1] xp, double[::1] out):
    """
    Evaluate a piecewise polynomial.

    Parameters
    ----------
    c : ndarray, shape (4, m+1,)
        Coefficients of the local polynomial of order 3 in m+1 intervals, including the two linear
        extrapolation intervals. Coefficient of highest order-term comes first.
    x : ndarray, shape (m,)
        Breakpoints of polynomials.
    xp : ndarray, shape (r,)
        Points to evaluate the piecewise polynomial at.
    out : ndarray, shape (r,)
        Value of the polynomial at each of the input points. This argument is modified in-place.
    """
    cdef int i, j, m
    cdef double xpi, dx
    m = x.shape[0]

    # Evaluate.
    for i in range(len(xp)):
        xpi = xp[i]
        j = find_interval(&x[0], m, xpi) # Find correct interval
        if 0 < j < m:
            dx = xpi - x[j - 1]
            out[i] = _evaluate(&c[j, 0], dx)
        elif j == 0:
            dx = xpi - x[0]
            out[i] = c[0, 2] * dx + c[0, 3]
        elif j == m:
            dx = xpi - x[m - 1]
            out[i] = c[m, 2] * dx + c[m, 3]
        else:
            out[i] = nan # xpi is nan etc
            continue


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def derivative(const double[:,::1] c, const double[::1] x, const double[::1] xp, double[::1] out):
    """
    Evaluate the first order derivative of a piecewise polynomial.

    Parameters
    ----------
    c : ndarray, shape (4, m+1,)
        Coefficients of the local polynomial of order 3 in m+1 intervals, including the two linear
        extrapolation intervals. Coefficient of highest order-term comes first.
    x : ndarray, shape (m,)
        Breakpoints of polynomials.
    xp : ndarray, shape (r,)
        Points to evaluate the derivative of piecewise polynomial at.
    out : ndarray, shape (r,)
        Value at each of the input points. This argument is modified in-place.
    """
    cdef int i, j, m
    cdef double xpi, dx
    m = x.shape[0]

    # Evaluate
    for i in range(len(xp)):
        xpi = xp[i]
        j = find_interval(&x[0], m, xpi) # Find correct interval
        if 0 < j < m:
            dx = xpi - x[j - 1]
            out[i] = _derivative(&c[j, 0], dx)
        elif j == 0:
            out[i] = c[0, 2]
        elif j == m:
            out[i] = c[m, 2]
        else:
            out[i] = nan # xpi is nan etc
            continue


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def solve(const double[:,::1] c, const double[::1] x, const double[::1] y, const double[::1] yp,
          double[::1] out):
    """
    Evaluate the inverse of a piecewise polynomial.

    Parameters
    ----------
    c : ndarray, shape (4, m+1,)
        Coefficients of the local polynomial of order 3 in m+1 intervals, including the two linear
        extrapolation intervals. Coefficient of highest order-term comes first.
    x : ndarray, shape (m,)
        Breakpoints of polynomials.
    y : ndarray, shape (m,)
        Value of polynomials at breakpoints.
    yp : ndarray, shape (r,)
        Points to evaluate the inverse of piecewise polynomial at.
    out : ndarray, shape (r,)
        Solution at each of the input points. This argument is modified in-place.
    """
    cdef int i, j, m
    cdef double ypi
    m = x.shape[0]

    # Evaluate
    for i in range(len(yp)):
        ypi = yp[i]
        j = find_interval(&y[0], m, ypi) # Find correct interval
        if 0 < j < m:
            out[i] = x[j - 1] + solve_bisect(&c[j, 0], ypi, x[j - 1], x[j])
        elif j == 0:
            out[i] = x[0] + (ypi - c[0, 3]) / c[0, 2]
        elif j == m:
            out[i] = x[m - 1] + (ypi - c[m, 3]) / c[m, 2]
        else:
            out[i] = nan # xval is nan etc
            continue


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def is_monotone(const double[:,::1] c, const double[::1] x, uint8[::1] out):
    cdef int i, m
    m = x.shape[0]
    for i in range(1, m):
        out[i - 1] = _is_monotone(&c[i, 0], 0., x[i] - x[i - 1])
