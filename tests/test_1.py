import numpy as np
from covar import covar_shrink

from rpy2.robjects import r
import rpy2.rinterface
from rpy2.robjects.functions import SignatureTranslatedFunction
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()


def setup():
    global corpcor
    try:
        corpcor = importr('corpcor')
    except rpy2.rinterface.RRuntimeError:
        r("install.packages('corpcor', repos='http://cran.us.r-project.org')")
        corpcor = importr('corpcor')


def test_1():
    for X in [np.random.randn(10,3), np.random.randn(100,3)]:
        r_result = corpcor.cov_shrink(X, lambda_var=0, verbose=False)
        py_result = covar_shrink(X)

        np.testing.assert_array_almost_equal(r_result, py_result[0])
        np.testing.assert_almost_equal(r.attr(r_result, 'lambda')[0], py_result[1])
