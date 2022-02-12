import numpy as np


def sampen(x, dim, r, scale=True):
    return entropy(x, dim, r, scale=scale)


def sampen_naive(x, m, r):
    return np.log(count_matches(x, m, r)/count_matches(x, m+1, r))


def count_matches(x, l, tol):
    n = x.size
    matches = 0
    for i in range(0, n-l):
        for j in range(i+1, n-l):
            if np.max(np.abs(x[i:i+l] - x[j:j+l])) <= tol:
                matches += 1
    return matches


def cross_sampen(x1, x2, dim, r, scale=True):
    return entropy([x1, x2], dim, r, scale)


def fuzzyen(x, dim, r, n, scale=True):
    return entropy(x, dim, r, n=n, scale=scale, remove_baseline=True)


def cross_fuzzyen(x1, x2, dim, r, n, scale=True):
    return entropy([x1, x2], dim, r, n, scale=scale, remove_baseline=True)


def pattern_mat(x, m):
    """
    Construct a matrix of `m`-length segments of `x`.

    Parameters
    ----------
    x : (N, ) array_like
        Array of input data.
    m : int
        Length of segment. Must be at least 1. In the case that `m` is 1, the
        input array is returned.

    Returns
    -------
    patterns : (m, N-m+1)
        Matrix whose first column is the first `m` elements of `x`, the second
        column is `x[1:m+1]`, etc.

    Examples
    --------
    >>> p = pattern_mat([1, 2, 3, 4, 5, 6, 7], 3])
    array([[ 1.,  2.,  3.,  4.,  5.],
           [ 2.,  3.,  4.,  5.,  6.],
           [ 3.,  4.,  5.,  6.,  7.]])
    """
    x = np.asarray(x).ravel()
    if m == 1:
        return x
    else:
        N = len(x)
        patterns = np.zeros((m, N-m+1))
        for i in range(m):
            patterns[i, :] = x[i:N-m+i+1]
        return patterns


def entropy(x, dim, r, n=1, scale=True, remove_baseline=False):
    """
    Calculate the entropy of a signal.

    Parameters
    ----------
    x : (N, ) array_like
        Input time series. May also be a length-2 list [x1, x2], in which case
        the cross entropy between x1 and x2 is calculated.
    dim : int
        Embedding dimension.
    r : float
        Tolerance (max absolute difference between segments) for SampEn or the
        width of the fuzzy exponential function for FuzzyEn. Larger `r` makes
        the function wider. Typical range is [0.2, 1].
    n : float, optional
        Step width of fuzzy exponential function for FuzzyEn (ignored
        otherwise). Larger `n` makes the function more rectangular. Usually in
        the range [1, 5] or so. Default is 1.
    scale : bool, optional
        If True, scale the data (zero mean, unit variance). Default is True.
    remove_baseline : bool, optional
        If True, remove the baseline from the pattern vectors. Used for
        FuzzyEn. Default is False (SampEn).

    Returns
    -------
    entropy : float
        The calculated entropy.
    """
    fuzzy = True if remove_baseline else False
    cross = True if isinstance(x, list) else False
    N = len(x[0]) if cross else len(x)
    npat = N - dim

    if scale:
        if cross:
            x = [_scale(np.copy(x[0])), _scale(np.copy(x[1]))]
        else:
            x = _scale(np.copy(x))

    phi = [0, 0]  # phi(m), phi(m+1)
    for j in [0, 1]:
        m = dim + j
        if cross:
            patterns = [pattern_mat(x[0], m)[:, :npat],
                        pattern_mat(x[1], m)[:, :npat]]
        else:
            patterns = pattern_mat(x, m)[:, :npat]

        if remove_baseline:
            if cross:
                patterns[0] = _remove_baseline(patterns[0], axis=0)
                patterns[1] = _remove_baseline(patterns[1], axis=0)
            else:
                patterns = _remove_baseline(patterns, axis=0)

        count = np.zeros(npat)
        for i in range(npat):
            if cross:
                if m == 1:
                    sub = patterns[1][i]
                else:
                    sub = patterns[1][:, [i]]
                dist = np.max(np.abs(patterns[0] - sub), axis=0)
            else:
                if m == 1:
                    sub = patterns[i]
                else:
                    sub = patterns[:, [i]]
                dist = np.max(np.abs(patterns - sub), axis=0)

            if fuzzy:
                sim = np.exp(-np.power(dist, n) / r)
            else:
                sim = dist <= r

            count[i] = np.sum(sim) - 1

        #phi[j] = np.mean(count) / (N - dim - 1)
        phi[j] = np.sum(count)

    return np.log(phi[0] / phi[1])


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
