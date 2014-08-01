import numpy as np


def center_kernel(K, copy=True):
    '''
    Centered version of a kernel matrix (corresponding to centering the)
    implicit feature map.
    '''
    means = K.mean(axis=0)
    if copy:
        K = K - means[None, :]
    else:
        K -= means[None, :]
    K -= means[:, None]
    K += means.mean()
    return K


def alignment(K1, K2):
    '''
    Returns the kernel alignment
        <K1, K2>_F / (||K1||_F ||K2||_F)
    defined by
        Cristianini, Shawe-Taylor, Elisseeff, and Kandola (2001).
        On Kernel-Target Alignment. NIPS.

    Note that the centered kernel alignment of
        Cortes, Mohri, and Rostamizadeh (2012).
        Algorithms for Learning Kernels Based on Centered Alignment. JMLR 13.
    is just this applied to center_kernel()s.
    '''
    return np.sum(K1 * K2) / np.linalg.norm(K1) / np.linalg.norm(K2)
