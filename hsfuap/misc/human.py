import humanize


def array_size(array, **k):
    return humanize.filesize.naturalsize(array.size * array.dtype.itemsize, **k)
