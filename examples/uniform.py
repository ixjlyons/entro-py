"""
Computes FuzzyEn of uniformly distributed random number sequences for different
values of fuzzy function width `r`. The result should look roughly linear.
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    import entropy
except:
    sys.path.insert(0, '..')
    import entropy


def main():
    N = 100
    rs = np.logspace(-3, 0, 10)

    fig, ax = plt.subplots()

    es = []
    for r in rs:
        runs = []
        for i in range(50):
            x = np.random.uniform(0, 1, N)
            runs.append(entropy.fuzzyen(x, 2, r, 2))
        es.append(np.mean(runs))

    ax.semilogx(rs, es, 'o')
    ax.set_ylim(0, 6)

    plt.show()


if __name__ == '__main__':
    main()
