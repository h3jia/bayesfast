cimport cython
import numpy as np
cimport numpy as np
ctypedef np.uint8_t uint8
from libc.math cimport log, exp


__all__ = ['_from_original_f', '_from_original_f2', '_to_original_f', 
           '_to_original_f2', '_to_original_j', '_to_original_j2', 
           '_to_original_jj', '_to_original_jj2']


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _from_original_f(const double[::1] x, const double[:, ::1] ranges, 
                     double[::1] out_f, const uint8[:, ::1] hard_bounds, 
                     const size_t n):
    cdef size_t i
    cdef double tmp
    for i in range(n):
        tmp = (x[i] - ranges[i, 0]) / (ranges[i, 1] - ranges[i, 0])
        if hard_bounds[i, 0] and hard_bounds[i, 1]:
            if tmp <= 0. or tmp >= 1.:
                raise ValueError('variable #{} out of bound.'.format(i))
            tmp = log(tmp / (1 - tmp))
        elif hard_bounds[i, 0] and (not hard_bounds[i, 1]):
            if tmp <= 0.:
                raise ValueError('variable #{} our of bound.'.format(i))
            tmp = log(tmp)
        elif (not hard_bounds[i, 0]) and hard_bounds[i, 1]:
            if tmp >= 1.:
                raise ValueError('variable #{} our of bound.'.format(i))
            tmp = log(1 - tmp)
        out_f[i] = tmp

        
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _from_original_f2(const double[:, ::1] x, const double[:, ::1] ranges, 
                      double[:, ::1] out_f, const uint8[:, ::1] hard_bounds, 
                      const size_t n, const size_t m):
    cdef size_t i
    for i in range(m):
        _from_original_f(x[i], ranges, out_f[i], hard_bounds, n)
        

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _to_original_f(const double[::1] x, const double[:, ::1] ranges, 
                   double[::1] out_f, const uint8[:, ::1] hard_bounds, 
                   const size_t n):
    cdef size_t i
    cdef double tmp
    for i in range(n):
        tmp = x[i]
        if hard_bounds[i, 0] and hard_bounds[i, 1]:
            tmp = 1 / (1 + exp(-tmp))
        elif hard_bounds[i, 0] and (not hard_bounds[i, 1]):
            tmp = exp(tmp)
        elif (not hard_bounds[i, 0]) and hard_bounds[i, 1]:
            tmp = 1 - exp(tmp)
        tmp = ranges[i, 0] + tmp * (ranges[i, 1] - ranges[i, 0])
        out_f[i] = tmp


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _to_original_f2(const double[:, ::1] x, const double[:, ::1] ranges, 
                    double[:, ::1] out_f, const uint8[:, ::1] hard_bounds, 
                    const size_t n, const size_t m):
    cdef size_t i
    for i in range(m):
        _to_original_f(x[i], ranges, out_f[i], hard_bounds, n)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _to_original_j(const double[::1] x, const double[:, ::1] ranges, 
                   double[::1] out_j, const uint8[:, ::1] hard_bounds, 
                   const size_t n):
    cdef size_t i
    cdef double tmp
    for i in range(n):
        tmp = x[i]
        if hard_bounds[i, 0] and hard_bounds[i, 1]:
            tmp = 1 / (1 + exp(-tmp))
            tmp = tmp * (1 - tmp)
        elif hard_bounds[i, 0] and (not hard_bounds[i, 1]):
            tmp = exp(tmp)
        elif (not hard_bounds[i, 0]) and hard_bounds[i, 1]:
            tmp = -exp(tmp)
        else:
            tmp = 1.
        tmp *= (ranges[i, 1] - ranges[i, 0])
        out_j[i] = tmp


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _to_original_j2(const double[:, ::1] x, const double[:, ::1] ranges, 
                    double[:, ::1] out_j, const uint8[:, ::1] hard_bounds, 
                    const size_t n, const size_t m):
    cdef size_t i
    for i in range(m):
        _to_original_j(x[i], ranges, out_j[i], hard_bounds, n)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _to_original_jj(const double[::1] x, const double[:, ::1] ranges, 
                    double[::1] out_j, const uint8[:, ::1] hard_bounds, 
                    const size_t n):
    cdef size_t i
    cdef double tmp, tmp2
    for i in range(n):
        tmp = x[i]
        if hard_bounds[i, 0] and hard_bounds[i, 1]:
            tmp2 = exp(tmp)
            tmp = -tmp2 * (tmp2 - 1) / (tmp2 + 1) / (tmp2 + 1) / (tmp2 + 1)
        elif hard_bounds[i, 0] and (not hard_bounds[i, 1]):
            tmp = exp(tmp)
        elif (not hard_bounds[i, 0]) and hard_bounds[i, 1]:
            tmp = -exp(tmp)
        else:
            tmp = 0.
        tmp *= (ranges[i, 1] - ranges[i, 0])
        out_j[i] = tmp


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _to_original_jj2(const double[:, ::1] x, const double[:, ::1] ranges, 
                     double[:, ::1] out_j, const uint8[:, ::1] hard_bounds, 
                     const size_t n, const size_t m):
    cdef size_t i
    for i in range(m):
        _to_original_jj(x[i], ranges, out_j[i], hard_bounds, n)
