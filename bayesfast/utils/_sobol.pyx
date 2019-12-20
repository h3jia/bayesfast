# Based on sobol.cc by F. Y. Kuo and S. Joe, rewritten with Cython by He Jia
# With original copyright notice as below
# URL: https://web.maths.unsw.edu.au/~fkuo/sobol/index.html
#
################################################################################
#
# Frances Y. Kuo
#
# Email: <f.kuo@unsw.edu.au>
# School of Mathematics and Statistics
# University of New South Wales
# Sydney NSW 2052, Australia
# 
# Last updated: 21 October 2008
#
#   You may incorporate this source code into your own program 
#   provided that you
#   1) acknowledge the copyright owner in your program and publication
#   2) notify the copyright owner by email
#   3) offer feedback regarding your experience with different direction numbers
#
#
# -----------------------------------------------------------------------------
# Licence pertaining to sobol.cc and the accompanying sets of direction numbers
# -----------------------------------------------------------------------------
# Copyright (c) 2008, Frances Y. Kuo and Stephen Joe
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
# 
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
# 
#     * Neither the names of the copyright holders nor the names of the
#       University of New South Wales and the University of Waikato
#       and its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
################################################################################

import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport ceil, log, pow
from libc.stdio cimport FILE, fopen, fgets, fscanf
from libc.stdlib cimport malloc, free

__all__ = ['_sobol']


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _sobol(const unsigned N, const unsigned D, const unsigned char[:] dir_file, 
           double[:, ::1] points):
    cdef:
        unsigned i, j, k, value, d, s, a, L
        unsigned *C, *V, *X, *m
        FILE *infile
        char buffer[1000] # used to skip the first line
    
    L = <unsigned>ceil(log(<double>N) / log(2.0)) # max number of bits needed
    infile = fopen(<char *>&dir_file[0], "r")
    fgets(buffer, 1000, infile)
    
    C = <unsigned *>malloc(N * sizeof(unsigned))
    V = <unsigned *>malloc((L + 1) * sizeof(unsigned))
    X = <unsigned *>malloc(N * sizeof(unsigned))
    
    try:
        # C[i] = index from the right of the first zero bit of i
        C[0] = 1
        for i in range(1, N):
            C[i] = 1
            value = i
            while value & 1:
                value >>= 1
                C[i] += 1

        # points[i][j] = the jth component of the ith point
        #                with i indexed from 0 to N-1 
        #                and j indexed from 0 to D-1
        for i in range(D):
            points[0, i] = 0

        # ----- Compute the first dimension -----

        # Compute direction numbers V[1] to V[L], scaled by pow(2,32)
        for i in range(1, L + 1):
            V[i] = 1 << (32 - i) # all m's = 1

        # Evalulate X[0] to X[N-1], scaled by pow(2,32)
        X[0] = 0
        for i in range(1, N):
            X[i] = X[i-1] ^ V[C[i-1]]
            points[i, 0] = <double>X[i] / pow(2.0, 32) # *** the actual points
            #         ^ 0 for first dimension 

        # ----- Compute the remaining dimensions -----

        for j in range(1, D):
            # Read in parameters from file 
            fscanf(infile, " %u %u %u", &d, &s, &a)
            try:
                m = <unsigned *>malloc((s + 1) * sizeof(unsigned))
                for i in range(1, s + 1):
                    fscanf(infile, " %u", &m[i])
                # Compute direction numbers V[1] to V[L], scaled by pow(2,32)
                if L <= s:
                    for i in range(1, L + 1):
                        V[i] = m[i] << (32 - i)
                else:
                    for i in range(1, s + 1):
                        V[i] = m[i] << (32 - i)
                    for i in range(s + 1, L + 1):
                        V[i] = V[i - s] ^ (V[i - s] >> s)
                        for k in range(1, s):
                            V[i] ^= (((a >> (s - 1 - k)) & 1) * V[i - k])
                # Evalulate X[0] to X[N-1], scaled by pow(2,32)
                X[0] = 0
                for i in range(1, N):
                    X[i] = X[i - 1] ^ V[C[i - 1]]
                    # *** the actual points
                    points[i, j] = <double>X[i] / pow(2.0, 32)
                    #         ^ j for dimension (j+1)
            finally:
                # Clean up
                free(m)
    finally:    
        # Clean up
        free(C)
        free(V)
        free(X)
