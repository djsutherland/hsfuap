#!/usr/bin/env python

import h5py
import sdm

def convert_file(inp, out, factors=[.25, 1, 4]):
    for div in inp.keys():
        if div == '_meta':
            continue
        for k in inp[div].keys():
            divs = inp[div][k][()]
            med = np.median(d[np.triu_indices_from(d)])

            for factor in factors:
                name = 'median * {}'.format(factor)
                print '/'.join(div, k, name)
                g = out.require_group(div).require_group(k)
                if name not in g:
                    g[name] = sdm.sdm.make_km(divs, med * factor)
                else:
                    print '\talready there'

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--factors', nargs='+', type=float, default=[.25, 1, 4])
    parser.add_argument('in_file')
    parser.add_argument('out_file')
    args = parser.parse_args()

    with h5py.File(args.in_file, 'r') as inp, h5py.File(args.out_file) as out:
        convert_file(inp, out, args.factors)

if __name__ == '__main__':
    main()
