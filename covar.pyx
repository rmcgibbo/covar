import numpy as np
cimport cython
from scipy.linalg.cython_blas cimport dgemm


def covar_shrink(const double[:, ::1] X, shrinkage=None):
    """Compute shrinkage estimates of covariance, respectively.

    Parameter
    ---------
    X : array, shape=(n, p)
        Data matrix. Each row represents a data point, and each column
        represents a feature.
    shrinkage : float, optional
        The covariance shrinkage intensity (range 0-1). If shrinkage is not
        specified (the default) it is estimated using an analytic formula
        from Schafer and Strimmer (2005). For shrinkage=0 the empirical
        correlations are recovered.

    Returns
    -------
    cov : array, shape=(p, p)
        Estimated covariance matrix of the data
    shrinkage : float
        The applied covariance shrinkage intensity.

    References
    ----------
    .. [1] Schafer, J., and K. Strimmer. 2005. A shrinkage approach to
        large-scale covariance estimation and implications for functional
        genomics. Statist. Appl. Genet. Mol. Biol. 4:32.

    Notes
    -----
    This shrinkage estimator corresponds to "Target D": (diagonal, unequal
    variance) as described in [1]. The estimator takes the form.

    .. math::
        \hat{\Sigma} = (1-\gamma) \Sigma_{sample} + \gamma T

    where :math:`\Sigma_{sample}` is the unbiased empirical covariance matrix,

    ..math::
        `(\Sigma_{sample})_{ij} = (1/(n-1)) \sum_{k}
            (x_{ki} - \bar{x}_i) * (x_{kj} - \bar{x}_j)`

    the matrix :math:`T` is the shrinkage target, and the scalar
    :math:`gamma \in [0, 1]` is the shrinkage intensity.

    """
    cdef int n, p, i, j, k
    n, p = X.shape[0], X.shape[1]

    cdef double gamma_num = 0
    cdef double gamma_den = 0
    cdef double s_ij, s_ii, gamma

    cdef double[::1] X_mean = np.mean(X, axis=0)
    cdef double[::1] X_std = np.std(X, axis=0)
    cdef double[:, ::1] X_meaned = np.empty_like(X)
    cdef double[:, ::1] w_ij_bar = np.zeros((p, p))
    cdef double[:, ::1] r = np.zeros((p, p))
    cdef double[:, ::1] var_r = np.zeros((p, p))
    cdef double[:, ::1] out = np.zeros((p, p))

    for i in range(n):
        for j in range(p):
            X_meaned[i, j] = X[i, j] - X_mean[j]

    cy_dgemm_TN(X_meaned, X_meaned, w_ij_bar, 1.0/n)

    if shrinkage is not None:
        gamma = float(shrinkage)
    else:
        for i in range(p):
            for j in range(p):
                r[i, j] = (n / ((n - 1.0) * X_std[i] * X_std[j])) * w_ij_bar[i, j]

        for k in range(n):
            for i in range(p):
                for j in range(p):
                    var_r[i,j] += (X_meaned[k,i]*X_meaned[k,j] - w_ij_bar[i,j])**2


        for i in range(p):
            for j in range(p):
                var_r[i,j] *= (n / ((n-1.0)**3 * X_std[i]*X_std[i]*X_std[j]*X_std[j]))


        for i in range(p):
            for j in range(p):
                if i != j:
                    gamma_num += var_r[i,j]
                    gamma_den += r[i,j]**2

        gamma =  np.clip(gamma_num / gamma_den, 0, 1)

    for i in range(p):
        for j in range(p):
            s_ij = (n / (n-1.0)) * w_ij_bar[i, j]
            out[i, j] = (1.0-gamma) * s_ij
            if i == j:
                out[i, i] += gamma * s_ij
        if out[i, j] == -0:
            out[i, j] = 0

    return np.asarray(out), gamma


@cython.boundscheck(False)
cdef inline int cy_dgemm_TN(double[:, ::1] a, double[:, ::1] b, double[:, ::1] c, double alpha=1.0, double beta=0.0) nogil:
    """C = beta*C + alpha*dot(A.T, B)
    """
    cdef int m, k, n
    m = a.shape[1]
    k = a.shape[0]
    n = b.shape[1]
    if a.shape[0] != b.shape[0] or a.shape[1] != c.shape[0] or b.shape[1] != c.shape[1]:
        return -1

    dgemm("N", "T", &n, &m, &k, &alpha, &b[0,0], &n, &a[0,0], &m, &beta, &c[0,0], &n)
    return 0
