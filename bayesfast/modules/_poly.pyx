cimport cython
from libc.stdlib cimport malloc, free
from cython.parallel import prange, parallel


__all__ = ['_quadratic_f', '_quadratic_j', '_cubic_2_f', '_cubic_2_j', 
           '_cubic_3_f', '_cubic_3_j', '_lsq_quadratic', '_lsq_cubic_2', 
           '_lsq_cubic_3', '_set_quadratic', '_set_cubic_2', '_set_cubic_3']


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _quadratic_f(const double[::1] x, const double[:, :, ::1] a,
                 double[::1] out, int m, int n):
    cdef size_t i, j, k
    cdef double *t = <double *> malloc(m * sizeof(double))
    if not t:
        raise MemoryError('cannot malloc required array in _quadratic_f.')
    try:
        for i in prange(m, nogil=True, schedule='static'):
            out[i] = 0.
            for j in range(n):
                t[i] = 0.
                for k in range(j, n):
                    t[i] += a[i, j, k] * x[k]
                out[i] += t[i] * x[j]
    finally:
        free(t)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _quadratic_j(const double[::1] x, const double[:, :, ::1] a,
                 double[:, ::1] out, int m, int n):
    cdef size_t i, j, k
    for i in prange(m, nogil=True, schedule='static'):
        for j in range(n):
            out[i, j] = 2 * a[i, j, j] * x[j]
            for k in range(j):
                out[i, j] += a[i, k, j] * x[k]
            for k in range(j + 1, n):
                out[i, j] += a[i, j, k] * x[k]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _cubic_2_f(const double[::1] x, const double[:, :, ::1] a, double[::1] out,
               int m, int n):
    cdef size_t i, j, k
    cdef double *t = <double *> malloc(m * sizeof(double))
    if not t:
        raise MemoryError('cannot malloc required array in _cubic_2_f.')
    try:
        for i in prange(m, nogil=True, schedule='static'):
            out[i] = 0.
            for j in range(n):
                t[i] = 0.
                for k in range(n):
                    t[i] += a[i, j, k] * x[k]
                out[i] += t[i] * x[j] * x[j]
    finally:
        free(t)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _cubic_2_j(const double[::1] x, const double[:, :, ::1] a,
               double[:, ::1] out, int m, int n):
    cdef size_t i, j, k
    for i in prange(m, nogil=True, schedule='static'):
        for j in range(n):
            out[i, j] = 0.
            for k in range(n):
                out[i, j] += a[i, j, k] * x[k]
            out[i, j] *= 2. * x[j]
            for k in range(n):
                out[i, j] += a[i, k, j] * x[k] * x[k]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _cubic_3_f(const double[::1] x, const double[:, :, :, ::1] a,
               double[::1] out, int m, int n):
    cdef size_t i, j, k, l
    cdef double *s = <double *> malloc(m * sizeof(double))
    cdef double *t = <double *> malloc(m * sizeof(double))
    if not (s and t):
        raise MemoryError('cannot malloc required array in _cubic_3_f.')
    try:
        for i in prange(m, nogil=True, schedule='static'):
            out[i] = 0.
            for j in range(n - 2):
                s[i] = 0.
                for k in range(j + 1, n - 1):
                    t[i] = 0.
                    for l in range(k + 1, n):
                        t[i] += a[i, j, k, l] * x[l]
                    s[i] += t[i] * x[k]
                out[i] += s[i] * x[j]
    finally:
        free(s)
        free(t)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _cubic_3_j(const double[::1] x, const double[:, :, :, ::1] a,
               double[:, ::1] out, int m, int n):
    cdef size_t i, j, k, l
    cdef double *t = <double *> malloc(m * sizeof(double))
    if not t:
        raise MemoryError('cannot malloc required array in _cubic_3_j.')
    try:
        for i in prange(m, nogil=True, schedule='static'):
            for j in range(n):
                out[i, j] = 0.
                for k in range(j):
                    t[i] = 0.
                    for l in range(k + 1, j):
                        t[i] += a[i, k, l, j] * x[l]
                    out[i, j] += t[i] * x[k]
                    t[i] = 0.
                    for l in range(j + 1, n):
                        t[i] += a[i, k, j, l] * x[l]
                    out[i, j] += t[i] * x[k]
                for k in range(j + 1, n):
                    t[i] = 0.
                    for l in range(k + 1, n):
                        t[i] += a[i, j, k, l] * x[l]
                    out[i, j] += t[i] * x[k]
    finally:
        free(t)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _lsq_quadratic(const double[:, ::1] x, double[:, ::1] out, int m, int n):
    cdef size_t i, j, k, l
    for i in range(m):
        j = 0
        for k in range(n):
            for l in range(k, n):
                out[i, j] = x[i, k] * x[i, l]
                j += 1


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _lsq_cubic_2(const double[:, ::1] x, double[:, ::1] out, int m, int n):
    cdef size_t i, j, k, l
    for i in range(m):
        j = 0
        for k in range(n):
            for l in range(n):
                out[i, j] = x[i, k] * x[i, k] * x[i, l]
                j += 1


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _lsq_cubic_3(const double[:, ::1] x, double[:, ::1] out, int m, int n):
    cdef size_t i, j, k, l, p
    for i in range(m):
        j = 0
        for k in range(n):
            for l in range(k + 1, n):
                for p in range(l + 1, n):
                    out[i, j] = x[i, k] * x[i, l] * x[i, p]
                    j += 1


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _set_quadratic(const double[::1] a, double[:, ::1] coef, int n):
    cdef size_t i, j, k
    i = 0
    for j in range(n):
        for k in range(j, n):
            coef[j, k] = a[i]
            i += 1


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _set_cubic_2(const double[::1] a, double[:, ::1] coef, int n):
    cdef size_t i, j, k
    i = 0
    for j in range(n):
        for k in range(n):
            coef[j, k] = a[i]
            i += 1


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _set_cubic_3(const double[::1] a, double[:, :, ::1] coef, int n):
    cdef size_t i, j, k, l
    i = 0
    for j in range(n):
        for k in range(j + 1, n):
            for l in range(k + 1, n):
                coef[j, k, l] = a[i]
                i += 1
