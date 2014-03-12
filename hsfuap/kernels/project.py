from __future__ import division

from functools import partial

import numpy as np
import scipy.linalg

from six.moves import xrange


def identity(x):
    return x


def symmetrize(mat, destroy=False):
    '''
    Returns the mean of mat and its transpose.

    If destroy, invalidates the passed-in matrix.
    '''
    # TODO: figure out a no-copy version of this that actually works...
    #       might have to write it in cython. probably not worth it.
    mat = mat + mat.T
    mat /= 2
    return mat


def _transformer(transform, test_matrix):
    '''
    Applies a given transformation matrix to the matrix of test vector
    similarities (num_test x num_train).
    '''
    return np.dot(transform, test_matrix.T).T


def project_psd(mat, min_eig=0, destroy=False, negatives_likely=True,
                ret_test_transformer=False):
    '''
    Project a real symmetric matrix to PSD by discarding any negative
    eigenvalues from its spectrum. Passing min_eig > 0 lets you similarly make
    it positive-definite, though this may not technically be a projection...?

    Symmetrizes the matrix before projecting.

    If destroy is True, invalidates the passed-in matrix.

    If negatives_likely (default), optimizes for the case where we expect there
    to be negative eigenvalues.

    If ret_test_transformer, also returns a function which takes a matrix of
    test similarities (num_test x num_train) and returns a matrix to make
    treatment consistent. Uses the method of
       Chen, Y., Garcia, E. K., Gupta, M. R., Rahimi, A., & Cazzanti, L. (2009).
       Similarity-based classification: Concepts and algorithms.
       Journal of Machine Learning Research, 10, 747-776.
    '''
    mat = symmetrize(mat, destroy=destroy)

    # TODO: be smart and only get negative eigs?
    vals, vecs = scipy.linalg.eigh(mat, overwrite_a=negatives_likely)
    vals = vals.reshape(-1, 1)

    if ret_test_transformer:
        clip = np.dot(vecs, (vals > 0) * vecs.T)
        transform = partial(_transformer, clip)

    if negatives_likely or vals[0, 0] < min_eig:
        del mat
        np.maximum(vals, min_eig, vals)  # update vals in-place
        mat = np.dot(vecs, vals.reshape(-1, 1) * vecs.T)
        if not ret_test_transformer:
            del vals, vecs
        mat = symmetrize(mat, destroy=True)  # should be symmetric, but do it
                                             # anyway for numerical reasons

    if ret_test_transformer:
        return mat, transform
    return mat


def shift_psd(mat, min_eig=0, destroy=False, negatives_likely=True,
              ret_test_transformer=False):
    '''
    Turn a real symmetric matrix to PSD by adding to its diagonal. Passing
    min_eig > 0 lets you make it positive-definite.

    Symmetrizes the matrix before doing so.

    If destroy is True, modifies the passed-in matrix in-place.

    Ignores the negatives_likely argument (just there for consistency).

    If ret_test_transformer, also returns a function which takes a matrix of
    test similarities (num_test x num_train) and returns a matrix to make
    treatment consistent. For the shift method, which only affects
    self-similarities, this is just the identity function.
    '''
    mat = symmetrize(mat, destroy=destroy)
    lo, = scipy.linalg.eigvalsh(mat, eigvals=(0, 0))
    diff = min_eig - lo
    if diff < 0:
        r = xrange(mat.shape[0])
        mat[r, r] += diff

    if ret_test_transformer:
        return mat, identity
    return mat


def flip_psd(mat, destroy=False, negatives_likely=True,
             ret_test_transformer=False):
    '''
    Turn a real symmetric matrix into PSD by flipping the sign of any negative
    eigenvalues in its spectrum.

    If destroy is True, invalidates the passed-in matrix.

    If negatives_likely (default), optimizes for the case where we expect there
    to be negative eigenvalues.

    If ret_test_transformer, also returns a function which takes a matrix of
    test similarities (num_test x num_train) and returns a matrix to make
    treatment consistent. Uses the method of
       Chen, Y., Garcia, E. K., Gupta, M. R., Rahimi, A., & Cazzanti, L. (2009).
       Similarity-based classification: Concepts and algorithms.
       Journal of Machine Learning Research, 10, 747-776.
    '''
    mat = symmetrize(mat, destroy=destroy)

    # TODO: be smart and only get negative eigs?
    vals, vecs = scipy.linalg.eigh(mat, overwrite_a=negatives_likely)
    vals = vals.reshape(-1, 1)

    if ret_test_transformer:
        flip = np.dot(vecs, np.sign(vals) * vecs.T)
        transform = partial(_transformer, flip)

    if negatives_likely or vals[0, 0] < 0:
        del mat
        np.abs(vals, vals)  # update vals in-place
        mat = np.dot(vecs, vals * vecs.T)
        del vals, vecs
        mat = symmetrize(mat, destroy=True)  # should be symmetric, but do it
                                             # anyway for numerical reasons
    if ret_test_transformer:
        return mat, transform
    return mat


def square_psd(mat, destroy=False, negatives_likely=True,
               ret_test_transformer=False):
    '''
    Turns a real matrix into a symmetric psd one through S -> S S^T. Equivalent
    to squaring the eigenvalues in a spectral decomposition, or to using the
    similarities to test points as features in a linear classifier.

    Ignores the destroy and negatives_likely arguments (just there for
    consistency).

    If ret_test_transformer, also returns a function which takes a matrix of
    test similarities (num_test x num_train) and returns a matrix to make
    treatment consistent.
    '''
    if ret_test_transformer:
        # TODO: do this like a linear operator, or using sims as features, or...
        raise NotImplementedError("not sure how to transform test samples here")
    return np.dot(mat, mat.T)


def identity_psd(mat, destroy=False, negatives_likely=True,
                 ret_test_transformer=False):
    if ret_test_transformer:
        return mat, np.eye(mat.shape[0])
    return mat


psdizers = {
    'project': project_psd,
    'clip': project_psd,
    'shift': shift_psd,
    'flip': flip_psd,
    'square': square_psd,
    'identity': identity_psd,
}


def rbf_kernelize(dists, sigma, square=True, destroy=False):
    '''
    Passes a distance matrix through an RBF kernel.

    If destroy, does it in-place.
    '''
    if destroy:
        km = dists
        if square:
            km **= 2
        km /= -2 * sigma ** 2
    elif square:
        km = dists ** 2
        km /= -2 * sigma ** 2
    else:
        km = dists / (-2 * sigma ** 2)

    np.exp(km, km)  # inplace
    return km


def make_km(divs, sigma, square=True, destroy=False, negatives_likely=True,
            method='project', ret_test_transformer=False):
    '''
    Passes a distance matrix through an RBF kernel of bandwidth sigma, and then
    ensures that it's PSD through `method` (see `psdizers`). Default: projects
    to the nearest PSD matrix by clipping any negative eigenvalues.

    If destroy, invalidates the data in divs.

    If negatives_likely (default), optimizes memory usage for the case where we
    expect there to be negative eigenvalues.
    '''
    return psdizers[method](
        rbf_kernelize(divs, sigma, square=square, destroy=destroy),
        destroy=True,
        negatives_likely=negatives_likely,
        ret_test_transformer=ret_test_transformer)


def get_squared_dists(K, destroy=False):
    row_norms = np.diagonal(K)
    if destroy:
        row_norms = row_norms.copy()  # some numpys have diagonal return a view
        K *= -2
    else:
        K = -2 * K
    K += row_norms[:, np.newaxis]
    K += row_norms[np.newaxis, :]
    return K
