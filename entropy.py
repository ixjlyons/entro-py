import numpy as np


def sampen(x, dim, r, scale=True):
    return entropy(x, dim, r, scale=scale)


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
        the function wider. Typical range 0.2 -- 1.
    n : float, optional
        Step width of fuzzy exponential function for FuzzyEn. Larger `n` makes
        the function more rectangular. Usually in the range 1 -- 5 or so.
        Default is 1.
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
    cross = True if type(x) == list else False
    N = len(x[0]) if cross else len(x)

    if scale:
        if cross:
            x = [_scale(x[0]), _scale(x[1])]
        else:
            x = _scale(x)

    phi = [0, 0]  # phi(m), phi(m+1)
    for j in [0, 1]:
        m = dim + j
        if cross:
            patterns = [pattern_mat(x[0], m), pattern_mat(x[1], m)]
        else:
            patterns = pattern_mat(x, m)

        if remove_baseline:
            if cross:
                patterns[0] -= np.mean(patterns[0], axis=0)
                patterns[1] -= np.mean(patterns[1], axis=0)
            else:
                patterns -= np.mean(patterns, axis=0)

        count = np.zeros(N-m)
        for i in range(N-m):
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

        phi[j] = np.mean(count) / (N-m-1)

    return np.log(phi[0] / phi[1])


def _scale(x):
    return (x - np.mean(x)) / np.std(x, ddof=1)
