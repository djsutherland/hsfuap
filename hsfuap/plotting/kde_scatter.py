import numpy as np
from scipy.stats import gaussian_kde


def kdescatter(xs, ys, **kwargs):
    import matplotlib.pyplot as plt

    kwargs.setdefault('linewidths', 0)
    kwargs.setdefault('s', 20)
    kwargs.setdefault('cmap', 'winter')

    dset = np.asarray([xs, ys])
    kde = gaussian_kde(dset)
    densities = kde.evaluate(dset)
    plt.scatter(xs, ys, c=densities, **kwargs)
