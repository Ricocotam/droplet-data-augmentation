import numpy as np
import scipy.ndimage as transform


def dropout(x, proportion, cval, *, proportion_generator=None):
    """Dropout pixels on an image. Drops are done on only one channel

    Args:
        x: The image from which dropouts are done. Modified.
        proportion: percentage of the image to drop. Total pixels dropped is
            `int(prop * np.prod(x.shape))`
        cval: which value dropped pixels take
        proportion_generator: Optionnal. Use a generator to get propotion
            value. This is useful if you use ImageDataGenerator from Keras.

    Returns:
        The image with pixels dropped out. Image is modified, not copied

    Notes:
        Image given is modified, not copied.
    """
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


def windowed_transformation(x, window_size, transformation, channel_last=True, **transformation_kwargs):
    temp = transformation(x, **transformation_kwargs)

    indices = _get_random_indices(x.shape, window_size, channel_last)
    if channel_last:
        x[np.ix_(*indices)] = temp[np.ix_(*indices)]
    else:
        x[np.ix_(*indices)] = temp[np.ix_(*indices)]

    return x


def gaussian(x, sigma, window_size, mode="reflect", cval=0, channel_last=True):
    """Apply a gaussian filter over all the image but replaces only a random
        window.

    Args:
        x: The image to transform, modified.
        sigma: Sigma to use for gaussian filtering
        window_size: Which window size you want to use to keep the transformation.
            Might be an int are a tuple of length `len(x.shape)-1` as we don't
            consider channels being spatial dimensions. Thus the droplet is applied
            over all channels
        mode: @see scipy.ndimage.gaussian_filter. Default is "reflect"
        cval: @see scipy.ndimage.gaussian_filter. Default is 0
        channel_last: True if you put channel dimension as last in x.shape.
            True if RGB is shape (n, m, 3), False is it is (3, n, m).

    Returns:
        The image given as input modified. A gaussian filter has been applied on a copy
        and then only a random selected window has been selected on the actual image
        from which we replaced pixels by the pixels of the filtered image over all channels.
    """
    return windowed_transformation(x, window_size, transform.gaussian_filter, channel_last,
                                   {sigma=sigma, mode=mode, cval=cval})


def rotate(x, angle, axes, window_size, mode="reflect", cval=0, channel_last=True):
    return windowed_transformation(x, window_size, transform.rotate, channel_last,
                                   {axes=axes, mode=mode, cval=cval})


def shift(x, shift, window_size, mode="reflect", cval=0, channel_last=True):
    return windowed_transformation(x, window_size, transform.shift, channel_last,
                                   {shift=shift, mode=mode, cval=cval})


def zoom(x, zoom, window_size, mode="reflect", cval=0, channel_last=True):
    return windowed_transformation(x, window_size, transform.zoom, channel_last,
                                   {zoom=zoom, mode=mode, cval=cval})


def _get_random_indices(shape, window_size, channel_last):
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
