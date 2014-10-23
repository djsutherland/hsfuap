import numpy as np

from sklearn.grid_search import GridSearchCV
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KernelDensity

__all__ = ['kdescatter']

def kdescatter(xs, ys, log_color=False, atol=1e-4, rtol=1e-4,
               n_jobs=1, n_samp_scaling=100, n_samp_tuning=1000, ax=None,
               **kwargs):
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt

    kwargs.setdefault('linewidths', 0)
    kwargs.setdefault('s', 20)
    kwargs.setdefault('cmap', 'winter')

    X = np.asarray([xs, ys]).T
    n = X.shape[0]
    samp_X = X[np.random.choice(n, min(n_samp_scaling, n), replace=False)]
    median_sqdist = np.median(euclidean_distances(samp_X, squared=True))
    bws = np.logspace(-2, 2, num=10) * np.sqrt(median_sqdist)
    est = GridSearchCV(KernelDensity(), {'bandwidth': bws}, n_jobs=n_jobs)
    est.fit(X[np.random.choice(n, min(n_samp_tuning, n), replace=False)])
    bw = est.best_params_['bandwidth']

    kde = KernelDensity(bandwidth=bw)
    kde.fit(X)
    densities = kde.score_samples(X)
    if not log_color:
        np.exp(densities, out=densities)
    ax.scatter(xs, ys, c=densities, **kwargs)
