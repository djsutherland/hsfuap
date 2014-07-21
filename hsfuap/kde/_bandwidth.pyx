cimport cython
from cython.parallel cimport prange, threadid
from cpython.exc cimport PyErr_CheckSignals

import numpy as np
cimport numpy as np
from scipy.linalg import solve_triangular, cholesky
from sklearn.metrics.pairwise import pairwise_kernels

@cython.boundscheck(False)
@cython.cdivision(True)
def next_C(float[:, ::1] feats, float[:, ::1] C, n_jobs=1):
    cdef int n = feats.shape[0], p = feats.shape[1]
    cdef int i, j, k, l, kl, tid
    cdef float[:, :] responsibility, new, C_chol
    
    C_chol = cholesky(C, lower=True)
    K = pairwise_kernels(
        solve_triangular(np.asarray(C_chol), np.asarray(feats).T, lower=True).T,
        metric='rbf', gamma=.5)
    # this doesn't include the 1/sqrt(|2 pi C|) factor, but it cancels anyway
    
    # turn into "responsibilities" over n
    np.fill_diagonal(K, 0)
    K /= K.sum(axis=1)[:, None] * n
    responsibility = K
    
    # TODO: would probably be more cache-efficient to transpose feats
    new = np.zeros_like(C)

    for kl in prange(p * p, nogil=True, schedule='static', num_threads=n_jobs):
        k = kl % p
        l = kl // p
        tid = threadid()

        for i in xrange(n):
            for j in xrange(n):
                if threadid() == 0:
                    with gil:
                        PyErr_CheckSignals()
                if j == i:
                    continue
                for k in xrange(p):
                    for l in xrange(p):
                        new[k, l] += (feats[i, k] - feats[j, k]) * (feats[i, l] - feats[j, l]) \
                                     * responsibility[i, j]
    return new
