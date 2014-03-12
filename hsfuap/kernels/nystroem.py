#!/usr/bin/env python
from __future__ import division, print_function

from functools import partial
import itertools
import random

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.metrics.pairwise import euclidean_distances

from ..misc import progress
from .project import get_squared_dists


################################################################################
### Nystroem framework

def nys_error(K, picked):
    assert picked.dtype.kind == 'b'
    A = K[np.ix_(picked, picked)]
    B = K[np.ix_(picked, ~picked)]
    C = K[np.ix_(~picked, ~picked)]

    A_pinv = np.linalg.pinv(A)

    Bhat = A.dot(A_pinv).dot(B)
    Berr = ((Bhat - B) ** 2).sum()

    Chat = B.T.dot(A_pinv.dot(B))
    Cerr = ((Chat - C) ** 2).sum()

    return np.sqrt(2 * Berr + Cerr)


def _run_nys(W, pick, start_n=5, max_n=None):
    # choose an initial couple of points uniformly at random
    N = W.shape[0]
    picked = np.zeros(W.shape[0], dtype=bool)
    picked[np.random.choice(W.shape[0], start_n, replace=False)] = True
    n = picked.sum()

    seen_pts = np.zeros((N, N), dtype=bool)
    seen_pts[picked, :] = True
    seen_pts[:, picked] = True

    if max_n is None:
        max_n = W.shape[0]

    n_picked = [n]
    n_evaled = [seen_pts.sum()]
    pickeds = [picked.copy()]
    rmse = [nys_error(W, picked)]

    # could do this faster with woodbury, probably
    pbar = progress(maxval=max_n).start()
    pbar.update(n)
    try:
        while n_picked[-1] < max_n:
            indices = pick(picked, seen_pts)
            picked[indices] = True
            n = picked.sum()

            seen_pts[picked, :] = True
            seen_pts[:, picked] = True

            n_picked.append(n)
            n_evaled.append(seen_pts.sum())
            pickeds.append(picked.copy())
            rmse.append(nys_error(W, picked))
            pbar.update(min(n, max_n))
    except Exception:
        import traceback
        traceback.print_exc()

    pbar.finish()
    return pd.DataFrame({
        'n_picked': n_picked,
        'n_evaled': n_evaled,
        'rmse': rmse,
        'picked': pickeds,
    })


def _run_nys_noniter(K, pick, start_n=5, max_n=None, step_size=1):
    N = K.shape[0]
    if max_n is None:
        max_n = N

    ns = np.arange(start_n, max_n + 1, step_size)

    rmses = []
    pickeds = []
    n_evaleds = []

    picked = np.zeros(N, dtype=bool)

    for n in progress()(ns):
        indices, n_evaled = pick(n)
        picked.fill(False)
        picked[indices] = True
        rmse = nys_error(K, picked)

        pickeds.append(picked.copy())
        n_evaleds.append(n_evaled)
        rmses.append(rmse)

    return pd.DataFrame({
        'n_picked': ns,
        'n_evaled': n_evaleds,
        'rmse': rmses,
        'picked': pickeds,
    })


def pick_up_to(ary, n, p=None):
    if np.shape(ary) == () and ary > 0:
        ary = np.arange(ary)
    if p is not None:
        n = min(n, np.nonzero(p)[0].size)
    return np.random.choice(ary, replace=False, size=min(n, ary.shape[0]), p=p)


################################################################################
### Fake method that gets the lower bound on reconstruction error

def run_lowerbound(K, start_n=5, max_n=None, step_size=1):
    N = K.shape[0]
    if max_n is None:
        max_n = N
    ns = np.arange(start_n, max_n + 1, step_size)

    # frobenius error of best rank-K reconstruction is
    # the L2 norm of the vector of K+1 and following singular values.
    svs = np.r_[np.linalg.svd(K, compute_uv=False), 0]
    all_rmses = np.sqrt(np.cumsum((svs ** 2)[::-1])[::-1])

    return pd.DataFrame({
        'n_picked': ns,
        'n_evaled': N ** 2,
        'rmse': all_rmses[ns],
    })


################################################################################
### Uniform

def run_uniform(W, start_n=5, max_n=None, step_size=1):
    def pick(picked, seen_pts):
        return pick_up_to((~picked).nonzero()[0], n=step_size)
    return _run_nys(W, pick, start_n=start_n, max_n=max_n)


################################################################################
### Deshpande-style adaptation

def run_adapt_full(W, start_n=5, max_n=None, step_size=1):
    def pick(picked, seen_pts):
        uc, _, _ = np.linalg.svd(W[:, picked], full_matrices=False)
        err = W - uc.dot(uc.T).dot(W)
        probs = np.zeros(picked.size)
        probs[~picked] = (err[~picked, :] ** 2).sum(axis=1)
        probs /= probs.sum()

        seen_pts.fill(True)
        return pick_up_to(probs.size, p=probs, n=step_size)
    return _run_nys(W, pick, start_n=start_n, max_n=max_n)


################################################################################
### Leverages-based stuff

def leverages_of_unknown(A, B, rcond=1e-15):
    # NOTE: definitely a better way to do this
    # can you compute  A^{-1/2} B  without eigendecomposing A?
    #   (is solve(sqrtm(A), B) better?)

    # get pinv(sqrtm(A))
    # assume that A is actually psd; any negative eigs are noise/numerical error
    A_vals, A_vecs = np.linalg.eigh(A)
    np.maximum(A_vals, 0, out=A_vals)
    np.sqrt(A_vals, out=A_vals)
    cutoff = np.max(A_vals) * rcond
    zeros = A_vals < cutoff
    A_vals[zeros] = 0
    A_vals[~zeros] **= -1
    inv_sqrt_A = np.dot(A_vecs, A_vals.reshape(-1, 1) * A_vecs.T)

    # better way to do this:
    #   x^T A^{-1} x = ||L \ x||^2 where L = chol(A)
    # can probably figure out how to use that here.
    X = inv_sqrt_A.dot(B)
    S = A + X.dot(X.T)
    Y = np.linalg.pinv(S)

    return np.einsum('ki,kl,li->i', X, Y, X)


def run_adapt_full_lev(W, start_n=5, max_n=None, step_size=1):
    def pick(picked, seen_pts):
        uc, _, _ = np.linalg.svd(W[:, picked], full_matrices=False)
        err = W - uc.dot(uc.T).dot(W)
        err_u, _, _ = np.linalg.svd(err)
        probs = np.zeros(picked.size)
        # using this rank for leverage scores is kind of arbitrary
        probs[~picked] = (err_u[~picked, :picked.sum()] ** 2).sum(axis=1)
        probs /= probs.sum()

        seen_pts.fill(True)
        return pick_up_to(probs.size, p=probs, n=step_size)
    return _run_nys(W, pick, start_n=start_n, max_n=max_n)


def run_leverage_full_iter(W, start_n=5, max_n=None, step_size=1):
    # NOTE: not quite the full leverage-based algorithm.
    # That chooses leverage scores for the final rank, where
    # this does it iteratively. Not clear how different those are.
    u, s, v = np.linalg.svd(W)
    n = W.shape[0]

    def pick_by_leverage(picked, seen_pts):
        levs = np.zeros(n)
        levs[~picked] = (u[~picked, :picked.sum()] ** 2).sum(axis=1)
        levs /= levs.sum()

        seen_pts.fill(True)
        return pick_up_to(levs.shape[0], p=levs, n=step_size)

    return _run_nys(W, pick_by_leverage, start_n=start_n, max_n=max_n)


def run_leverage_est(W, start_n=5, max_n=None, step_size=1):
    # Like above, but pick based on the leverage scores of \hat{W} instead
    # of W, so it doesn't use any knowledge we don't have.
    def pick_by_leverage(picked, seen_pts):
        levs = leverages_of_unknown(W[np.ix_(picked, picked)],
                                    W[np.ix_(picked, ~picked)])
        assert np.all(np.isfinite(levs))
        dist = levs / levs.sum()
        unpicked_idx = pick_up_to(dist.shape[0], p=dist, n=step_size)
        return (~picked).nonzero()[0][unpicked_idx]

    return _run_nys(W, pick_by_leverage, start_n=start_n, max_n=max_n)


################################################################################
### Determinant-based stuff

# Here, we use the block LU decomposition:
#  [A, B; C, D] = [I, 0; C*inv(A), I]
#                 [A, 0; 0, D - C inv(A) B]
#                 [I, inv(A) B; 0, I]
#
# So
# det([K, v; v', k]) = det([I, 0; v' inv(K), 1])
#                      det([K, 0; 0, k - v' inv(K) v])
#                      det([I, inv(K) v; 0, 1])
#                    = (k - v' inv(K) v) det(K)
# (and you get the same thing if you try to decompose the other way).
#
# So, we plug in K = K(I, I), v = K(I, j), k = K(j, j),
# and compute v' inv(K) v as ||chol(K) \ v||^2.
def pick_det_greedy(picked, seen_pts, K, samp):
    # TODO: could build this up across runs blockwise
    chol_K_picked = linalg.cholesky(K[np.ix_(picked, picked)], lower=True)
    # det_K_picked = np.prod(np.diagonal(chol_K_picked)) ** 2
    #   don't need this: it's the same for everyone

    dets = np.zeros(picked.size)
    tmps = linalg.solve_triangular(
        chol_K_picked, K[np.ix_(picked, ~picked)], lower=True)
    dets[~picked] = K[~picked, ~picked] - np.sum(tmps ** 2, axis=0)
    #  * det_K_picked

    if samp:
        return pick_up_to(picked.size, p=dets / dets.sum(), n=1)
    else:
        return np.argmax(dets)


def run_determinant_greedy_samp(K, start_n=5, max_n=None, step_size=1):
    assert step_size == 1
    f = partial(pick_det_greedy, K=K, samp=True)
    return _run_nys(K, f, start_n=start_n, max_n=max_n)


def run_determinant_greedy(K, start_n=5, max_n=None, step_size=1):
    assert step_size == 1
    f = partial(pick_det_greedy, K=K, samp=False)
    return _run_nys(K, f, start_n=start_n, max_n=max_n)


def rejection_sample_det(K, n, max_samps=None):
    # algorithm from section 6.2.1 of Arcolano (2011)
    # in the case where we assume that the diagonal is unity
    # TODO: account for non-one diagonals

    assert np.allclose(np.diagonal(K), 1)
    N = K.shape[0]

    _log_betas = np.empty(N)
    _log_betas.fill(np.nan)
    def log_beta(i):
        if np.isnan(_log_betas[i]):
            sgn, _log_betas[i] = np.linalg.slogdet(K[i:i + n, i:i + n])
            assert sgn == 1
        return _log_betas[i]

    it = itertools.count() if max_samps is None else xrange(max_samps)
    for n_rejects in it:
        proposal = np.random.choice(N, n, replace=False)
        sgn, logdet = np.linalg.slogdet(K[np.ix_(proposal, proposal)])
        assert sgn == 1
        log_prob = logdet - log_beta(proposal.min())
        if np.random.binomial(1, p=np.exp(log_prob)):
            return proposal

    raise ValueError("Didn't accept a sample in {} tries.".format(max_samps))


def metropolis_sample_det(K, n, num_iter):
    N = K.shape[0]

    curr = np.random.choice(N, n, replace=False)
    proposal = curr.copy()
    all_inds = set(xrange(N))

    sgn, curr_logdet = np.linalg.slogdet(K[np.ix_(curr, curr)])
    assert sgn == 1

    choice = np.random.choice

    for _ in xrange(num_iter):
        i_ind, = choice(n)
        j, = random.sample(all_inds.difference(proposal), 1)
        proposal[i_ind] = j

        sgn, prop_logdet = np.linalg.slogdet(K[np.ix_(proposal, proposal)])
        assert sgn == 1

        if random.random() < np.exp(prop_logdet - curr_logdet):
            # accept the proposal
            curr[i_ind] = j
        else:
            # make proposal equal to curr again
            proposal[i_ind] = curr[i_ind]

    return curr


################################################################################
### K-means

def pick_kmeans(x, n):
    # NOTE: doesn't make sense to do this iteratively
    # run k-means clustering
    from vlfeat import vl_kmeans
    centers = vl_kmeans(x, num_centers=n)

    # pick points closest to the cluster centers
    from cyflann import FLANNIndex
    picked = FLANNIndex().nn(x, centers, num_neighbors=1)[0]

    N = x.shape[0]
    return picked, N ** 2 - (N - n) ** 2


def run_kmeans(K, X, start_n=5, max_n=None, step_size=1):
    return _run_nys_noniter(K, partial(pick_kmeans, X),
                            start_n=start_n, max_n=max_n, step_size=step_size)


################################################################################
### Kernel k-means

def pick_kernel_kmeans(K, n):
    from .kmeans import KernelKMeans
    km = KernelKMeans(n_clusters=n, kernel='precomputed')
    km.fit(K)

    # find the point closest to each cluster center in kernel space
    dists = np.zeros((K.shape[0], n))
    km._compute_dist(K, dists, km.within_distances_, update_within=False)
    picked = dists.argmin(axis=0)

    N = K.shape[0]
    return picked, N ** 2


def run_kernel_kmeans(K, start_n=5, max_n=None, step_size=1):
    return _run_nys_noniter(K, partial(pick_kmeans, K),
                            start_n=start_n, max_n=max_n, step_size=step_size)


################################################################################
### Pick according to the kmeans++ initialization criterion


def init_kmeanspp(sq_dists, n):
    N = sq_dists.shape[0]
    picked = np.zeros(N, dtype=bool)
    idx = np.random.choice(N)
    picked[idx] = True

    to_center = sq_dists[idx, :]

    for i in xrange(1, n):
        idx = np.random.choice(N, p=to_center / to_center.sum())
        picked[idx] = True
        np.minimum(to_center, sq_dists[idx, :], out=to_center)

    return picked


def kmeanspp_init_picker(sq_dists):
    N = sq_dists.shape[0]
    have_filled = [False]

    def pick(picked, seen_pts):
        if not have_filled[0]:
            seen_pts.fill(True)
            have_filled[0] = True

        to_center = sq_dists[picked, :].min(axis=0)
        to_center /= to_center.sum()
        return np.random.choice(N, p=to_center)
    return pick


def run_kmeanspp_initonly(K, X, step_size=1, **kwargs):
    assert step_size == 1
    sq_dists = euclidean_distances(X, squared=True)
    return _run_nys(K, kmeanspp_init_picker(sq_dists), **kwargs)


def run_kernel_kmeanspp_initonly(K, step_size=1, **kwargs):
    assert step_size == 1
    sq_dists = get_squared_dists(K)
    return _run_nys(K, kmeanspp_init_picker(sq_dists), **kwargs)


################################################################################
### Step-wise kmeans++

# TODO: kernel version
def run_kmeanspp_stepwise(K, X, step_size=1, **kwargs):
    assert step_size == 1  # TODO: support having more than one new point

    N = X.shape[0]
    sq_norms = np.einsum('ij,ij->i', X, X)
    sq_dist_to_X = partial(
        euclidean_distances, Y=X, Y_norm_squared=sq_norms, squared=True)
    sq_dists = sq_dist_to_X(X)

    have_filled = [False]
    def pick(picked, seen_pts):
        if not have_filled[0]:
            seen_pts.fill(True)
            have_filled[0] = True

        # ids of, distances to nearest fixed center
        dist_to_fixed = sq_dists[picked, :]
        nearest_fixed = dist_to_fixed.argmin(axis=0)
        to_nearest_fixed = dist_to_fixed[nearest_fixed, xrange(N)]
        del dist_to_fixed

        # pick an initial new center according to kmeans++
        idx = np.random.choice(N, p=to_nearest_fixed / to_nearest_fixed.sum())
        new = X[idx]

        # run k-means algorithm with all but one point fixed
        old_new_is_better = True
        new_is_better = False
        while not np.all(old_new_is_better == new_is_better):
            # distances to the means
            dist_to_new = sq_dist_to_X(new).ravel()
            new_is_better = dist_to_new < to_nearest_fixed

            # reassign means
            new = X[new_is_better].mean()

            old_new_is_better = new_is_better

        if not np.any(new_is_better):
            print("WARNING: kmeanspp_stepwise: choosing randomly")
            unpicked = ~picked
            return np.random.choice(N, p=unpicked / unpicked.sum())

        # pick the closest point to the new mean
        return dist_to_new.argmin()

    return _run_nys(K, pick, **kwargs)


################################################################################
### SMGA

def _do_nys(K, picked, out):
    notpicked = ~picked
    A = K[np.ix_(picked, picked)]
    B = K[np.ix_(picked, notpicked)]

    out[np.ix_(picked, picked)] = A
    out[np.ix_(picked, notpicked)] = B
    out[np.ix_(notpicked, picked)] = B.T
    out[np.ix_(notpicked, notpicked)] = B.T.dot(np.linalg.pinv(A).dot(B))


def run_smga_frob(K, start_n=5, max_n=None, eval_size=59, step_size=1):
    assert step_size == 1

    # choose an initial couple of points uniformly at random
    N = K.shape[0]
    if max_n is None:
        max_n = N

    picked = np.zeros(N, dtype=bool)
    picked[np.random.choice(N, start_n, replace=False)] = True

    # TODO: do version of algorithm that estimates error rather than getting the
    #       full thing.
    seen_pts = np.ones((N, N), dtype=bool)
    seen_pts[picked, :] = True
    seen_pts[:, picked] = True

    n = picked.sum()
    n_picked = [n]
    n_evaled = [seen_pts.sum()]

    est = np.empty_like(K)
    err = np.empty_like(K)
    _do_nys(K, picked, est)
    np.subtract(K, est, out=err)
    rmse = [np.linalg.norm(err, 'fro')]
    pickeds = [picked.copy()]

    err_prods = np.empty_like(err)

    pbar = progress(maxval=max_n).start()
    pbar.update(n)

    try:
        while n_picked[-1] < max_n:
            pool = pick_up_to((~picked).nonzero()[0], n=eval_size)

            np.dot(err, err, out=err_prods)  # each entry is  err[i].dot(err[j])
            imp_factors = np.array([
                (err_prods[i, :] ** 2).sum() / (err[i] ** 2).sum()
                for i in pool
            ])
            i = pool[np.argmax(imp_factors)]

            picked[i] = True
            n_picked.append(picked.sum())
            n_evaled.append(seen_pts.sum())
            pickeds.append(picked.copy())

            _do_nys(K, picked, est)
            np.subtract(K, est, out=err)
            rmse.append(np.linalg.norm(err, 'fro'))

            pbar.update(n_picked[-1])
    except Exception:
        import traceback
        traceback.print_exc()

    pbar.finish()
    return pd.DataFrame({
        'n_picked': n_picked,
        'n_evaled': [N ** 2] * len(n_picked),
        'rmse': rmse,
        'picked': pickeds,
    })


################################################################################

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-n', type=int, default=5)
    parser.add_argument('--max-n', type=int, default=None)
    parser.add_argument('--step-size', type=int, default=1)
    parser.add_argument('--method', '-m', required=True)
    parser.add_argument('--kernel-file', '-k', required=True)
    parser.add_argument('--kernel-path', '-K')
    parser.add_argument('--feats-path')
    parser.add_argument('outfile')
    args = parser.parse_args()

    method = globals()['run_{}'.format(args.method)]
    if args.method == 'kmeans':
        with np.load(args.feats_path) as d:
            X = np.sqrt(d['hists'])
        method = partial(method, X=X)

    if args.kernel_path:
        import h5py
        with h5py.File(args.kernel_file, 'r') as f:
            kernel = f[args.kernel_path][()]
    else:
        kernel = np.load(args.kernel_file)

    n, m = kernel.shape
    assert n == m

    d = method(kernel, start_n=args.start_n, max_n=args.max_n,
               step_size=args.step_size)
    d.to_csv(args.outfile)


if __name__ == '__main__':
    main()
