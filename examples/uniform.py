"""
=================================================
Entropy of Uniformly Distributed Random Sequences
=================================================


"""
print(__doc__)

import sys
import numpy as np
import matplotlib.pyplot as plt

try:
    import entropy
    import sampen
except:
    sys.path.insert(0, '.')
    import entropy
    import sampen


def main():
    N = 1000
    rep = 10
    rs = np.logspace(-2, 0, 20)

    fig, ax = plt.subplots()
    x = np.random.uniform(0, 1, size=(rep, N))

    es = []
    es_samp = []
    for r in rs:
        runs = []
        runs_samp = []
        for i in range(rep):
            runs.append(sampen.apen(x[i], 2, r))
            runs_samp.append(sampen.sampen(x[i], 2, r))
        es.append(np.mean(runs))
        es_samp.append(np.mean(runs_samp))

    ax.semilogx(rs, es, 'o', label='ApEn')
    ax.semilogx(rs, es_samp, 's', label='SampEn')
    ax.set_ylim(0, 6)
    ax.legend()

    plt.show()


if __name__ == '__main__':
    main()
