#!/usr/bin/env python
import pandas as pd
import numpy as np


def gather(fnames):
    info = []
    for fn in fnames:
        d = np.load(fn)

        right = d['preds'] == d['labels']
        fold_accs = np.array([right[test].mean() for train, test in d['folds']])
        info.append({
            'div_func': d['div_func'][()],
            'K': d['K'][()],
            'accuracy': d['accuracy'][()],
            'fold_accs': fold_accs,
        })
    return pd.DataFrame(info).set_index(['div_func', 'K']).sort_index()


def plot_results(info, ax=None):
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()

    for div_func, vals in info.groupby(level='div_func'):
        vals = vals.reset_index()
        ax.plot(vals.K, vals.accuracy, label=div_func, marker='o')
    ax.legend(loc='lower right')
    return ax


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('results', nargs='+')

    parser.add_argument('--save', default=None)

    g = parser.add_mutually_exclusive_group()
    g.add_argument('--plot', action='store_true', default=True)
    g.add_argument('--plot-to', dest='plot')
    g.add_argument('--no-plot', dest='plot', action='store_false')
    args = parser.parse_args()

    info = gather(args.results)

    if args.save:
        info.to_csv(args.save)

    if args.plot:
        if args.plot is not True:
            import matplotlib
            matplotlib.use('Agg')

        ax = plot_results(info)

        import matplotlib.pyplot as plt
        if args.plot is True:
            plt.show()
        else:
            plt.savefig(args.plot)
main()
