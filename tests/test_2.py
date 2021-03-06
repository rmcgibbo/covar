from __future__ import division
import os.path
import numpy as np
import scipy.stats
import matplotlib
matplotlib.use('Agg')
matplotlib.rc('font',family='serif')
import matplotlib.pyplot as plt
from covar import cov_shrink_ss, cov_shrink_rblw

DIRNAME = os.path.dirname(os.path.realpath(__file__))


def test_1():
    random = np.random.RandomState(0)
    p = 100
    sigma = scipy.stats.wishart(scale=np.eye(p), seed=random).rvs()
    Ns = [int(x) for x in [p/10, p/2, 2*p, 10*p]]
    x = np.arange(p)

    plt.figure(figsize=(8,8))

    for i, N in enumerate(Ns):
        X = random.multivariate_normal(mean=np.zeros(p), cov=sigma, size=N)
        S1 = np.cov(X.T)
        S2 = cov_shrink_ss(X)[0]
        S3 = cov_shrink_rblw(np.cov(X.T), len(X))[0]

        plt.subplot(3,2,i+1)
        plt.title('p/n = %.1f' % (p/N))

        plt.plot(x, sorted(np.linalg.eigvalsh(S2), reverse=True), 'b',  lw=2, label='cov_shrink_ss')
        plt.plot(x, sorted(np.linalg.eigvalsh(S3), reverse=True), 'g', alpha=0.7, lw=2, label='cov_shrink_rblw')
        plt.plot(x, sorted(np.linalg.eigvalsh(sigma), reverse=True), 'k--', lw=2, label='true')
        plt.plot(x, sorted(np.linalg.eigvalsh(S1), reverse=True), 'r--', lw=2, label='sample covariance')

        if i == 1:
            plt.legend(fontsize=10)

        # plt.ylim(max(plt.ylim()[0], 1e-4), plt.ylim()[1])
        plt.figtext(.05, .05,
"""Ordered eigenvalues of the sample covariance matrix (red),
cov_shrink_ss()-estimated covariance matrix (blue),
cov_shrink_rblw()-estimated covariance matrix (green), and
true  eigenvalues (dashed black). The data generated by sampling
from a p-variate normal distribution for p=100 and various
ratios of p/n. Note that for the larger value of p/n, the
cov_shrink_rblw() estimator is identical to the sample
covariance matrix.""")

        # plt.yscale('log')
        plt.ylabel('Eigenvalue')

    plt.tight_layout()
    plt.savefig('%s/test_2.png' % DIRNAME, dpi=300)



