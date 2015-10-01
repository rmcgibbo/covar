from __future__ import division
import os.path
import numpy as np
import scipy.stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from covar import cov_shrink
# from sklearn.covariance import ledoit_wolf, oas

DIRNAME = os.path.dirname(os.path.realpath(__file__))


def test_1():
    random = np.random.RandomState(0)
    p = 100
    sigma = scipy.stats.wishart(scale=np.eye(p), seed=random).rvs()
    Ns = [int(x) for x in [p/10, p/2, 2*p, 10*p]]
    x = np.arange(p)

    for i, N in enumerate(Ns):
        X = random.multivariate_normal(mean=np.zeros(p), cov=sigma, size=N)
        S1 = np.cov(X.T)
        S2 = cov_shrink(X)[0]
        # S3 = oas(X)[0]

        plt.subplot(2,2,i+1)
        plt.title('p/n = %.1f' % (p/N))
        plt.plot(x, sorted(np.linalg.eigvals(S1), reverse=True), 'r', lw=2)
        plt.plot(x, sorted(np.linalg.eigvals(S2), reverse=True), 'b', lw=2)
        # plt.plot(x, sorted(np.linalg.eigvals(S3), reverse=True), 'g', lw=2)

        plt.plot(x, sorted(np.linalg.eigvals(sigma), reverse=True), 'k--')
        plt.ylabel('Eigenvalue')

    plt.tight_layout()
    plt.savefig('%s/test_2.png' % DIRNAME, dpi=300)



