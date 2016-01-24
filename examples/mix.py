"""
Calculates the cross-FuzzyEn between two different pairs of MIX(p) processes
across exponential function widths. The intention is to replicate Figure 3a of
Xie et al. 2010 ("Cross-Fuzzy Entropy: A New Method to Test Pattern Synchrony
of Bivariate Time Series").
"""

import sys
import numpy as np
from numpy.random import uniform, binomial
import matplotlib.pyplot as plt

try:
    import entropy
except:
    sys.path.insert(0, '..')
    import entropy


alpha_m12 = np.power(np.sum(np.power(np.sin(2*np.pi*np.arange(1, 13)/12), 2))/12, -0.5)


def mix(p, size=None):
    """
    Generate MIX(p) random processes [1].

    Parameters
    ----------
    p : float
       Randomness of the process. Technically, it is the probability of drawing
       a sample from a uniform distribution rather than a deterministic
       sinusoid. If 0, a sinusoid is generated.
    size : int or tuple of ints, optional
        Output shape. Default is None, in which case a single value is
        returned.

    References
    ----------
    .. [1] `S. M. Pincus, "Approximate Entropy as a Measure of System
        Complexity," Proceedings of the National Academy of Sciences of the
        United States of America, vol. 88, no. 6, 1991.`
    """
    # deterministic sinusoidal signal
    x = alpha_m12 * np.sin(2*np.pi*np.arange(0, size[1])/12)
    # uniformly disributed samples
    y = uniform(-np.sqrt(3), np.sqrt(3), size)
    # bernoulli trials to select between deterministic/uniform samples
    z = binomial(1, p, size)
    m = np.multiply((1 - z), x) + np.multiply(z, y)
    return m


def main():
    N = 100
    rep = 10
    rs = np.logspace(-2, 0, 30)

    x_p2 = mix(0.2, (rep, N))
    x_p3 = mix(0.3, (rep, N))
    x_p5 = mix(0.5, (rep, N))

    es = np.zeros((2, len(rs)))
    for ir, r in enumerate(rs):
        runs = np.zeros((2, rep))
        for i in range(rep):
            runs[0, i] = entropy.cross_fuzzyen(x_p2[i, :], x_p3[i, :], 2, r, 2)
            runs[1, i] = entropy.cross_fuzzyen(x_p3[i, :], x_p5[i, :], 2, r, 2)

        es[:, ir] = np.mean(runs, axis=1)

    fig, ax = plt.subplots()
    ax.semilogx(rs, es[0], 'ro', label='MIX(0.2) vs. MIX(0.3)')
    ax.semilogx(rs, es[1], 'ko', label='MIX(0.3) vs. MIX(0.5)')
    ax.legend()

    plt.show()


if __name__ == '__main__':
    main()
