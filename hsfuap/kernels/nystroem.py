#!/usr/bin/env python
from __future__ import division

from functools import partial

import numpy as np
import pandas as pd

from ..misc import progress


def leverages_of_unknown(A, B, rcond=1e-15):
    # TODO: definitely a better way to do this
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

    X = inv_sqrt_A.dot(B)
    S = A + X.dot(X.T)
    Y = np.linalg.pinv(S)

    return np.einsum('ki,kl,li->i', X, Y, X)


def nys_error(K, picked):
    A = K[np.ix_(picked, picked)]
    B = K[np.ix_(picked, ~picked)]
    C = K[np.ix_(~picked, ~picked)]

    A_pinv = np.linalg.pinv(A)

    Bhat = A.dot(A_pinv).dot(B)
    Berr = ((Bhat - B) ** 2).sum()

    Chat = B.T.dot(A_pinv.dot(B))
    Cerr = ((Chat - C) ** 2).sum()

    return np.sqrt(2 * Berr + Cerr)


def _run_nys(W, pick, start_n=5):
    # choose an initial couple of points uniformly at random
    picked = np.zeros(W.shape[0], dtype='bool')
    picked[np.random.choice(W.shape[0], start_n, replace=False)] = True
    n = picked.sum()

    n_picked = [n]
    n_evaled = [n]
    rmse = [nys_error(W, picked)]

    # could do this faster with woodbury, probably
    pbar = progress(maxval=picked.size).start()
    pbar.update(n)
    try:
        while not picked.all():
            indices, extra_evaled = pick(picked)
            picked[indices] = True
            n = picked.sum()

            n_picked.append(n)
            n_evaled.append(extra_evaled.imag if np.iscomplex(extra_evaled)
                            else n + extra_evaled)
            rmse.append(nys_error(W, picked))
            pbar.update(n)
    except Exception as e:
        import traceback
        traceback.print_exc()

    pbar.finish()
    return pd.DataFrame(
        {'n_picked': n_picked, 'n_evaled': n_evaled, 'rmse': rmse})

def pick_up_to(ary, n, p=None):
    if np.shape(ary) == () and ary > 0:
        ary = np.arange(ary)
    if p is not None:
        n = min(n, np.nonzero(p)[0].size)
    return np.random.choice(ary, replace=False, size=min(n, ary.shape[0]), p=p)


def run_uniform(W, start_n=5, step_size=1):
    return _run_nys(
        W,
        lambda picked: (pick_up_to((~picked).nonzero()[0], n=step_size), 0),
        start_n=start_n)


def run_adapt_full(W, start_n=5, step_size=1):
    n = W.shape[0]
    def pick(picked):
        uc, _, _ = np.linalg.svd(W[:, picked], full_matrices=False)
        err = W - uc.dot(uc.T).dot(W)
        probs = np.zeros(picked.size)
        probs[~picked] = (err[~picked, :] ** 2).sum(axis=1)
        probs /= probs.sum()
        return (pick_up_to(probs.size, p=probs, n=step_size), n*1j)
    return _run_nys(W, pick, start_n=start_n)


def run_leverage_full_iter(W, start_n=5, step_size=1):
    # NOTE: not quite the full leverage-based algorithm.
    # That chooses leverage scores for the final rank, where
    # this does it iteratively. Not clear how different those are.
    u, s, v = np.linalg.svd(W)
    n = W.shape[0]

    def pick_by_leverage(picked):
        levs = np.zeros(n)
        levs[~picked] = (u[~picked, :picked.sum()] ** 2).sum(axis=1)
        levs /= levs.sum()
        return (pick_up_to(levs.shape[0], p=levs, n=step_size), n*1j)

    return _run_nys(W, pick_by_leverage, start_n=start_n)

def run_leverage_est(W, start_n=5, step_size=1):
    # Like above, but pick based on the leverage scores of \hat{W} instead
    # of W, so it doesn't use any knowledge we don't have.
    def pick_by_leverage(picked):
        levs = leverages_of_unknown(W[np.ix_(picked, picked)],
                                    W[np.ix_(picked, ~picked)])
        assert np.all(np.isfinite(levs))
        dist = levs / levs.sum()
        unpicked_idx = pick_up_to(dist.shape[0], p=dist, n=step_size)
        return ((~picked).nonzero()[0][unpicked_idx], 0)

    return _run_nys(W, pick_by_leverage, start_n=start_n)


def nys_kmeans(K, x, n):
    # NOTE: doesn't make sense to do this iteratively
    from vlfeat import vl_kmeans
    from cyflann import FLANNIndex
    centers = vl_kmeans(x, num_centers=n)
    picked = FLANNIndex().nn(x, centers, num_neighbors=1)[0]
    return nys_error(K, picked)

def run_kmeans(K, X, start_n=5, step_size=1):
    # NOTE: not actually iterative, unlike the others
    ns = range(start_n, K.shape[0] + 1, step_size)
    rmses = [nys_kmeans(K, X, n) for n in progress()(ns)]
    return pd.DataFrame({'n_picked': ns, 'n_evaled': ns, 'rmse': rmses})



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-n', type=int, default=5)
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

    d = method(kernel, start_n=args.start_n, step_size=args.step_size)
    d.to_csv(args.outfile)

if __name__ == '__main__':
    main()
