# Implements the algorithm of 
# opt2008.kyb.tuebingen.mpg.de/papers/leiva.pdf

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from sklearn.metrics.pairwise import pairwise_kernels

from ._bandwidth import next_C

def precompute_outers(feats):
    diffs = feats[:, None, :] - feats[None, :, :]
    return diffs[:, :, :, None] * diffs[:, :, None, :]


def next_C_pre(feats, outers, C, n_jobs=1):
    n, p = feats.shape
    
    # get all the kernel entries
    # this doesn't include the 1/sqrt(|2 pi C|) factor, but it cancels anyway
    C_chol = cholesky(C, lower=True)
    G = pairwise_kernels(
        solve_triangular(C_chol, feats.T, lower=True).T,
        metric='rbf', gamma=.5, n_jobs=n_jobs)    
    
    # reweight kernels into "responisibilities" over n
    np.fill_diagonal(G, 0)
    G /= G.sum(axis=1)[:, None] * n
    
    # new[k, l] = 1/n sum_i (sum_{j \ne i} outers[i, j, k, l] * G[i, j]) / (sum_{j \ne i} G[i, j])
    # becomes     sum_i sum_j outers[i, j, k, l] * G[i, j]   since G[i, i] = 0
    #           = einsum('ijkl,ij', outers, G)
    
    return np.einsum('ijkl,ij', outers, G)
