`covar`: shrinkage covariance estimation
=======================================

This Python package contains a single function, cov_shrink` which implements
a plug-in shrinkage estimator for the covariance matrix.

The estimator is described by [Schafer and Strimmer (2005)](http://www.degruyter.com/view/j/sagmb.2005.4.1/sagmb.2005.4.1.1175/sagmb.2005.4.1.1175.xml>),
where it is called "Target D: (diagonal, unequal variance)".

See the [documentation](https://pythonhosted.org/covar/) for more details.

### Installation

```
pip install covar
```

### Dependencies
1. Python (2.7, or 3.3+)
2. numpy
3. scipy (0.16+)
4. cython
