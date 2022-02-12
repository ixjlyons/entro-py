import numpy as np
from numpy.random import uniform, binomial
import sampen
import matplotlib.pyplot as plt


alpha_m12 = np.power(
    np.sum(np.power(np.sin(2*np.pi*np.arange(1, 13)/12), 2))/12,
    -0.5)


def mix(p, size=None):
    """Generate MIX(p) random processes.

    See [1] for a description of the process.

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
    .. [1] S. M. Pincus, "Approximate Entropy as a Measure of System
        Complexity," Proceedings of the National Academy of Sciences of the
        United States of America, vol. 88, no. 6, 1991.
    """
    # deterministic sinusoidal signal
    x = alpha_m12 * np.sin(2*np.pi*np.arange(0, size[1])/12)
    # uniformly disributed samples
    y = uniform(-np.sqrt(3), np.sqrt(3), size)
    # bernoulli trials to select between deterministic/uniform samples
    z = binomial(1, p, size)
    m = np.multiply((1 - z), x) + np.multiply(z, y)
    return m



N = 100
rep = 100
rs = np.logspace(-2, 0, 20)
x = mix(0.1, size=(rep, N))

fig, ax = plt.subplots()

es = []
es_samp = []
for r in rs:
    runs = []
    runs_samp = []
    for i in range(rep):
        runs.append(sampen.fuzzyen(x[i], 2, r, n=2))
        runs_samp.append(sampen.sampen(x[i], 2, r))
    es.append(np.mean(runs))
    es_samp.append(np.mean(runs_samp))

ax.semilogx(rs, es, 'o', label='fuzzyen')
ax.semilogx(rs, es_samp, 's', label='sampen')
ax.set_ylim(0, 6)
ax.legend()

plt.show()
