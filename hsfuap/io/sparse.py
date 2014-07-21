from scipy import sparse

def store_sparse(mat, hdf_group):
    if isinstance(mat, sparse.csr.csr_matrix):
        hdf_group['__type__'] = 'csr'
        for par in ['data', 'indices', 'indptr', 'shape']:
            hdf_group[par] = getattr(mat, par)
    else:
        raise TypeError("store_sparse only works for CSR matrices")


def load_sparse(hdf_group):
    if hdf_group['__type__'][()] == 'csr':
        return sparse.csr_matrix(
            (hdf_group['data'][()],
             hdf_group['indices'][()], hdf_group['indptr'][()]),
            shape=hdf_group['shape'][()],
        )
    else:
        raise TypeError("load_sparse only works for CSR matrices")
