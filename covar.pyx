import numpy as np
cimport cython
from scipy.linalg.cython_blas cimport dgemm


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cov_shrink_ss(const double[:, ::1] X, shrinkage=None):
    r"""Compute a shrinkage estimate of the covariance matrix using
    the Schafer and Strimmer (2005) method.

    Parameters
    ----------
    X : array, shape=(n, p)
        Data matrix. Each row represents a data point, and each column
        represents a feature.
    shrinkage : float, optional
        The covariance shrinkage intensity (range 0-1). If shrinkage is not
        specified (the default) it is estimated using an analytic formula
        from Schafer and Strimmer (2005). For ``shrinkage=0`` the empirical
        correlations are recovered.

    Returns
    -------
    cov : array, shape=(p, p)
        Estimated covariance matrix of the data.
    shrinkage : float
        The applied covariance shrinkage intensity.

    References
    ----------
    .. [1] Schafer, J., and K. Strimmer. 2005. A shrinkage approach to
        large-scale covariance estimation and implications for functional
        genomics. Statist. Appl. Genet. Mol. Biol. 4:32.
        http://doi.org/10.2202/1544-6115.1175

    Notes
    -----
    This shrinkage estimator corresponds to "Target D": (diagonal, unequal
    variance) as described in [1]. The estimator takes the form

    .. math::
        \hat{\Sigma} = (1-\gamma) \Sigma_{sample} + \gamma T,

    where :math:`\Sigma^{sample}` is the (noisy but unbiased) empirical
    covariance matrix,

    .. math::
        \Sigma^{sample}_{ij} = \frac{1}{n-1} \sum_{k=1}^n
            (x_{ki} - \bar{x}_i)(x_{kj} - \bar{x}_j),

    the matrix :math:`T` is the shrinkage target, a less noisy but biased
    estimator for the covariance, and the scalar :math:`\gamma \in [0, 1]` is
    the shrinkage intensity (regularization strength). This approaches uses a
    diagonal shrinkage target, :math:`T`:

    .. math::
        T_{ij} = \begin{cases}
            \Sigma^{sample}_{ii} &\text{ if } i = j\\
            0 &\text{ otherwise},
        \end{cases}

    The idea is that by taking a weighted average of these two estimators, we
    can get a combined estimator which is more accurate than either is
    individually, especially when :math:`p` is large. The optimal weighting,
    :math:`\gamma`, is determined **automatically** by minimizing the mean
    squared error. See [1] for details on how this can be done. The formula
    for :math:`\gamma` is

    .. math::
        \gamma = \frac{\sum_{i \neq j} \hat{Var}(r_{ij})}{\sum_{i \neq j} r^2_{ij}}

    where :math:`r` is the sample correlation matrix,

    .. math::
        r_{ij} = \frac{\Sigma^{sample}_{ij}}{\sigma_i \sigma_j},

    and :math:`\hat{Var}(r_{ij})` is given by

    .. math::
        \hat{Var}(r_{ij}) = \frac{n}{(n-1)^3 \sigma_i^2 \sigma_j^2} \sum_{k=1}^n
            (w_{kij} - \bar{w}_{ij})^2,

    with :math:`w_{kij} = (x_{ki} - \bar{x}_i)(x_{kj} - \bar{x}_j)`, and
    :math:`\bar{w}_{ij} = \frac{1}{n}\sum_{k=1}^n w_{kij}`.

    This method is equivalent to the ``cov.shrink`` method in the R package
    ``corpcor``, if the argument ``lambda.var`` is set to ``0``. See
    https://cran.r-project.org/web/packages/corpcor/ for details.

    See Also
    --------
    cov_shrink_rblw : similar method, using a different shrinkage target,
        :math:`T`.
    sklearn.covariance.ledoit_wolf : very similar approach, but uses a different
        shrinkage target, :math:`T`.
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
        gamma = max(0.0, min(1.0, float(shrinkage)))
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

        gamma =  max(0, min(1, gamma_num / gamma_den))

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
@cython.wraparound(False)
@cython.cdivision(True)
def cov_shrink_rblw(const double[:, ::1] S, int n, shrinkage=None):
    r"""Compute a shrinkage estimate of the covariance matrix using
    the Rao-Blackwellized Ledoit-Wolf estimator described by Chen et al.

    Parameters
    ----------
    S : array, shape=(n, n)
        Sample covariance matrix (e.g. estimated with np.cov(X.T))
    n : int
        Number of data points used in the estimate of S.
    shrinkage : float, optional
        The covariance shrinkage intensity (range 0-1). If shrinkage is not
        specified (the default) it is estimated using an analytic formula
        from Chen et al. (2009).

    Returns
    -------
    sigma : array, shape=(p, p)
        Estimated shrunk covariance matrix
    shrinkage : float
        The applied covariance shrinkage intensity.

    Notes
    -----
    This shrinkage estimator takes the form

    .. math::
        \hat{\Sigma} = (1-\gamma) \Sigma_{sample} + \gamma T

    where :math:`\Sigma^{sample}` is the (noisy but unbiased) empirical
    covariance matrix,

    .. math::
        \Sigma^{sample}_{ij} = \frac{1}{n-1} \sum_{k=1}^n
            (x_{ki} - \bar{x}_i)(x_{kj} - \bar{x}_j),

    the matrix :math:`T` is the shrinkage target, a less noisy but biased
    estimator for the covariance, and the scalar :math:`\gamma \in [0, 1]` is
    the shrinkage intensity (regularization strength). This approaches uses a
    scaled identity target, :math:`T`:

    .. math::
        T = \frac{\mathrm{Tr}(S)}{p} I_p

    The shrinkage intensity, :math:`\gamma`, is determined using the RBLW
    estimator from [2]. The formula for :math:`\gamma` is

    .. math::
        \gamma = \min(\alpha + \frac{\beta}{U})

    where :math:`\alpha`, :math:`\beta`, and :math:`U` are

    .. math::
        \alpha &= \frac{n-2}{n(n+2)} \\
        \beta  &= \frac{(p+1)n - 2}{n(n+2)} \\
        U      &= \frac{p\, \mathrm{Tr}(S^2)}{\mathrm{Tr}^2(S)} - 1

    One particularly useful property of this estimator is that it's **very
    fast**, because it doesn't require access to the data matrix at all (unlike
    :func:`cov_shrink_ss`). It only requires the sample covariance matrix
    and the number of data points `n`, as sufficient statistics.

    For reference, note that [2] defines another estimator, called the oracle
    approximating shrinkage estimator (OAS), but makes some mathematical errors
    during the derivation, and futhermore their example code published with
    the paper does not implement the proposed formulas.

    References
    ----------
    .. [2] Chen, Yilun, Ami Wiesel, and Alfred O. Hero III. "Shrinkage
       estimation of high dimensional covariance matrices" ICASSP (2009)
       http://doi.org/10.1109/ICASSP.2009.4960239

    See Also
    --------
    cov_shrink_ss : similar method, using a different shrinkage target, :math:`T`.
    sklearn.covariance.ledoit_wolf : very similar approach using the same
        shrinkage target, :math:`T`, but a different method for estimating the
        shrinkage intensity, :math:`gamma`.

    """
    cdef int i, j
    cdef int p = S.shape[0]
    if S.shape[1] != p:
        raise ValueError('S must be a (p x p) matrix')

    cdef double alpha = (n-2)/(n*(n+2))
    cdef double beta = ((p+1)*n - 2) / (n*(n+2))

    cdef double trace_S       # np.trace(S)
    cdef double trace_S2 = 0  # np.trace(S.dot(S))
    for i in range(p):
        trace_S += S[i,i]
        for j in range(p):
            trace_S2 += S[i,j]*S[i,j]

    cdef double U = ((p * trace_S2 / (trace_S*trace_S)) - 1)
    cdef double rho = min(alpha + beta/U, 1)

    F = (trace_S / p) * np.eye(p)
    return (1-rho)*np.asarray(S) + rho*F, rho



#############################  Private utilities #############################

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
