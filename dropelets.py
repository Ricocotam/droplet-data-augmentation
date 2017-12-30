import numpy as np
import scipy.ndimage as transform


def dropout(x, proportion, cval, *, proportion_generator=None):
    if proportion_generator:
        prop = next(proportion_generator)
    else:
        prop = proportion

    nb = int(prop * np.prod(x.shape))
    indices = []
    for l in x.shape:
        indices.append(np.random.randint(0, l, size=nb))
    x[indices] = cval
    return x


def gaussian(x, sigma, window_size, mode="reflect", cval=0, channel_last=True):
    # Gaussian filter is done over all channels
    # Replacement is also done over all channels

    # Transformation
    temp = transform.gaussian_filter(x, sigma, mode=mode, cval=cval)

    indices = get_random_indices(x.shape, window_size, channel_last)
    if channel_last:
        x[np.ix_(*indices)] = temp[np.ix_(*indices)]
    else:
        x[np.ix_(*indices)] = temp[np.ix_(*indices)]

    return x


def rotate(x, angle, axes, window_size, mode="reflect", cval=0, channel_last=True):
    # Axes -> read scipy.ndimage.rotate doc

    # Transformation
    temp = transform.rotate(x, angle, axes, mode=mode, cval=cval)

    indices = get_random_indices(x.shape, window_size, channel_last)
    if channel_last:
        x[np.ix_(*indices)] = temp[np.ix_(*indices)]
    else:
        x[np.ix_(*indices)] = temp[np.ix_(*indices)]

    return x


def shift(x, shift, window_size, mode="reflect", cval=0, channel_last=True):
    # shift -> read scipy.ndimage.shift doc
    temp = transform.shift(x, shift, mode=mode, cval=cval)

    indices = get_random_indices(x.shape, window_size, channel_last)
    if channel_last:
        x[np.ix_(*indices)] = temp[np.ix_(*indices)]
    else:
        x[np.ix_(*indices)] = temp[np.ix_(*indices)]

    return x


def zoom(x, zoom, window_size, mode="reflect", cval=0, channel_last=True):
    # zoom -> read scipy.ndimage.zoom doc
    temp = transform.rotate(x, zoom, mode=mode, cval=cval)

    indices = get_random_indices(x.shape, window_size, channel_last)
    if channel_last:
        x[np.ix_(*indices)] = temp[np.ix_(*indices)]
    else:
        x[np.ix_(*indices)] = temp[np.ix_(*indices)]

    return x


def get_random_indices(shape, window_size, channel_last):
    if type(window_size) is int:
        window_size = (window_size,) * (len(shape) - 1)

    if len(window_size) != len(shape) - 1:
        raise ValueError("window_size should be integer or tuple of same length of x spatial dimensions." +
                         "but we got window_size length= " + str(window_size) + " and spatial dimension length= " + str(len(shape) - 1))

    if channel_last:
        dims = shape[:-1]
        window_size = window_size + (shape[-1],)
    else:
        dims = shape[1:]
        window_size = (shape[0],) + window_size

    start_indices = []
    for i, l in enumerate(dims):
        start_indices.append(np.random.randint(0, l-window_size[i]))

    indices = [np.arange(start, start + wsize) for start, wsize in zip(start_indices, window_size)]
    if channel_last:
        indices.append(np.arange(shape[-1]))
    else:
        indices = [np.arange(shape[0])] + indices

    return indices
