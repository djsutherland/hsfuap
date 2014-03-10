"""Kernel K-means"""

# Original author: Mathieu Blondel <mathieu@mblondel.org>
# Modifications by: Dougal Sutherland <dougal@gmail.com>
# License: BSD 3 clause

import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_random_state


class KernelKMeans(BaseEstimator, ClusterMixin):
    """
    Kernel K-means

    Reference
    ---------
    Kernel k-means, Spectral Clustering and Normalized Cuts.
    Inderjit S. Dhillon, Yuqiang Guan, Brian Kulis.
    KDD 2004.
    """

    def __init__(self, n_clusters=3, max_iter=50, tol=1e-3, random_state=None,
                 kernel="linear", gamma=None, degree=3, coef0=1,
                 kernel_params=None, verbose=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.verbose = verbose

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)

    def fit(self, X, y=None, sample_weight=None):
        n_samples = X.shape[0]

        K = self._get_kernel(X)

        if sample_weight is None:
            self.sample_weight_ = np.ones(n_samples)
        else:
            self.sample_weight_ = sample_weight

        rs = check_random_state(self.random_state)
        self.labels_ = rs.randint(self.n_clusters, size=n_samples)
        # TODO: kmeans++ initialization?

        dist = np.zeros((n_samples, self.n_clusters))
        self.within_distances_ = np.zeros(self.n_clusters)

        cutoff = (1 - self.tol) * n_samples

        for it in xrange(self.max_iter):
            dist.fill(0)
            self._compute_dist(K, dist, self.within_distances_,
                               update_within=True)
            labels_old = self.labels_
            self.labels_ = dist.argmin(axis=1)

            # Compute the number of samples whose cluster did not change
            # since last iteration.
            n_same = np.sum(self.labels_ == labels_old)
            if n_same > cutoff:
                if self.verbose:
                    print "Converged at iteration", it + 1
                break

        self.X_fit_ = X
        return self

    def _compute_dist(self, K, dist, within_distances, update_within):
        """Compute a n_samples x n_clusters distance matrix using the
        kernel trick."""
        sw = self.sample_weight_

        for j in xrange(self.n_clusters):
            mask = self.labels_ == j
            wts = sw[mask]

            if np.sum(mask) == 0:
                raise ValueError("Empty cluster found, try smaller n_cluster.")

            denom = wts.sum()
            denomsq = denom * denom

            if update_within:
                wt_KK = K[np.ix_(mask, mask)] * wts[:, np.newaxis]
                wt_KK *= wts[np.newaxis, :]

                dist_j = np.sum(wt_KK) / denomsq
                within_distances[j] = dist_j
                dist[:, j] += dist_j
            else:
                dist[:, j] += within_distances[j]

            dist[:, j] -= 2 * np.sum(wts * K[:, mask], axis=1) / denom

    def predict(self, X):
        K = self._get_kernel(X, self.X_fit_)
        n_samples = X.shape[0]
        dist = np.zeros((n_samples, self.n_clusters))
        self._compute_dist(K, dist, self.within_distances_,
                           update_within=False)
        return dist.argmin(axis=1)


if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from sklearn.metrics import adjusted_rand_score

    X, y = make_blobs(n_samples=1000, centers=5, random_state=0)

    km = KernelKMeans(n_clusters=5, max_iter=100, random_state=0, verbose=1)
    print km.fit_predict(X)[:10]
    print km.predict(X[:10])
    print adjusted_rand_score(y, km.predict(X))
