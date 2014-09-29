import numpy as np

from sklearn.grid_search import GridSearchCV
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KernelDensity

__all__ = ['kdescatter']

def kdescatter(xs, ys, log_color=False, atol=1e-4, rtol=1e-4,
               n_jobs=1, n_samp_scaling=100, ax=None, **kwargs):
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt

    kwargs.setdefault('linewidths', 0)
    kwargs.setdefault('s', 20)
    kwargs.setdefault('cmap', 'winter')

    X = np.asarray([xs, ys]).T
    samp_X = X[np.random.choice(X.shape[0], n_samp_scaling, replace=False)]
    median_sqdist = np.median(euclidean_distances(samp_X, squared=True))
    bws = np.logspace(-2, 2, num=10) * np.sqrt(median_sqdist)
    est = GridSearchCV(KernelDensity(), {'bandwidth': bws}, n_jobs=n_jobs)
    est.fit(X)

    densities = est.best_estimator_.score_samples(X)
    if not log_color:
        np.exp(densities, out=densities)
    ax.scatter(xs, ys, c=densities, **kwargs)
