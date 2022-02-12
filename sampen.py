import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform



def sampen(x, m, r):
    """Sample entropy as defined by Richman and Moorman 2000.

    Consider a time series of length :math:`N`:

    ..math::

        x = \\left[ x_1, x_2, \\dots \\x_N \\right]

    :math:`B` is the total number of length-m subsequence matches within
    :math:`x`, where a match is determined by a threshold on the infinity norm
    of the difference between the subsequences. :math:`A` is the same for
    subsequences of length :math:`m+1`.
    """
    x = _scale(np.copy(x))
    templ = rolling_window(x, m + 1)
    A = np.sum(pdist(templ, 'chebyshev') <= r)
    B = np.sum(pdist(templ[:, :-1], 'chebyshev') <= r)
    return -np.log(A / B)


def apen(x, m, r):
    x  = _scale(np.copy(x))
    N = len(x)

    phi = [0, 0]
    for li, l in enumerate([m, m + 1]):
        templ = rolling_window(x, l)
        dists = squareform(pdist(templ, 'chebyshev'))
        cmi = np.mean(dists <= r, axis=1)
        phi[li] = np.mean(np.log(cmi))

    return phi[0] - phi[1]


def fuzzyen(x, m, r, n=1):
    x = _scale(np.copy(x))
    N = len(x)

    x_m = rolling_window(x, m)[:-1]
    x_m = _remove_baseline(x_m, axis=1)
    dists_m = pdist(x_m, 'chebyshev')
    sim_m = squareform(np.exp(-(dists_m**n) / r))
    phi_m = np.sum(sim_m) / (N - m) / (N - m - 1)

    x_mp1 = rolling_window(x, m + 1)
    x_mp1 = _remove_baseline(x_mp1, axis=1)
    dists_mp1 = pdist(x_mp1, 'chebyshev')
    sim_mp1 = squareform(np.exp(-(dists_mp1**n) / r))
    phi_mp1 = np.sum(sim_mp1) / (N - m) / (N - m - 1)

    return np.log(phi_m) - np.log(phi_mp1)


def _scale(x, axis=None):
    """
    Scales data by removing the mean and scaling to unit variance.

    Note: operates on the array passed in. Does not copy.
    """
    x = _remove_baseline(x, axis=axis)
    x /= np.std(x, ddof=1, axis=axis, keepdims=True)
    return x


def _remove_baseline(x, axis=None):
    """
    Removes the mean from the input data.

    Note: operates on the array passed in. Does not copy.
    """
    x -= np.mean(x, axis=axis, keepdims=True)
    return x


def rolling_window(array, n):
    """Create a rolling window from an array.

    An extra axis is added to efficiently compute statistics over. Use
    ``axis=-1`` to remove the extra axis.

    Parameters
    ----------
    array : ndarray
        The input array.
    n : int
        Window length.

    Returns
    -------
    window : array
        The length-n windows of the input array.

    Examples
    --------
    >>> import numpy as np
    >>> from axopy.features.util import rolling_window
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> rolling_window(x, 2)
    array([[1, 2],
           [2, 3],
           [3, 4],
           [4, 5]])
    >>> rolling_window(x, 3)
    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])
    >>> x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    >>> rolling_window(x, 2)
    array([[[1, 2],
            [2, 3],
            [3, 4]],
    <BLANKLINE>
           [[5, 6],
            [6, 7],
            [7, 8]]])

    References
    ----------
    .. [1] https://mail.scipy.org/pipermail/numpy-discussion/2010-December/054392.html # noqa
    """
    shape = array.shape[:-1] + (array.shape[-1] - n + 1, n)
    strides = array.strides + (array.strides[-1],)
    return np.lib.stride_tricks.as_strided(array,
                                           shape=shape,
                                           strides=strides)
