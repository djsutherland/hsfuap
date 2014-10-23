#!/usr/bin/env python

import h5py
import numpy as np
import sdm

def convert_file(in_file, out_file, factors=[.25, 1, 4]):
    with h5py.File(in_file, 'r') as inp:
        func_ks = [
            (df, k)
            for df, g in inp.iteritems() if df != '_meta'
            for k in g.iterkeys()
        ]

    meds = {}
    for df, k in func_ks:
        with h5py.File(in_file, 'r') as inp:
            divs = inp[df][k][()]

        if df in meds:
            med = meds[df]
        else:
            meds[df] = med = np.median(divs[np.triu_indices_from(divs)])

        for factor in factors:
            name = 'median * {}'.format(factor)
            print '/'.join((df, k, name))

            with h5py.File(out_file) as out:
                g = out.require_group(df).require_group(k)
                if name in g:
                    print '\talready there'
                    continue

            km = sdm.sdm.make_km(divs, med * factor)
            with h5py.File(out_file) as out:
                out[df][k][name] = km


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--factors', nargs='+', type=float, default=[.25, 1, 4])
    parser.add_argument('in_file')
    parser.add_argument('out_file')
    args = parser.parse_args()

    convert_file(args.in_file, args.out_file, args.factors)

if __name__ == '__main__':
    main()
