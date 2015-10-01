.. currentmodule:: covar

Covar: shrinkage covariance estimation
=======================================

This Python package contains a single function, :func:`cov_shrink` which implements a plug-in shrinkage estimator for the covariance matrix.

The estimator is described by `Schafer and Strimmer (2005) <http://www.degruyter.com/view/j/sagmb.2005.4.1/sagmb.2005.4.1.1175/sagmb.2005.4.1.1175.xml>`_, where it is
called "Target D: (diagonal, unequal variance)". See the paper and / or the function docstring for details.

.. figure:: /../tests/test_2.png
    :width: 500

    Ordered eigenvalues of the sample covariance matrix (red), :func:`cov_shrink`-estimated covariance matrix (blue), and true eigenvalues (dashed black) for simulated data with an underlying :math:`p`-variate normal distribution for p=100 and various rations of :math:`p/n`. This is a replication of Fig 1. of Schafer and Strimmer (2005).


This estimator is very similar to the ``ledoit_wolf`` and ``oas`` covariance matrix estimators implemented in ``sklearn.covariance``. Whereas those estimators shrink towards a diagonal matrix with equal entries along the diagonal, this shrinks towards an diagonal matrix whose diagonal entries are
identitical to those in the sample covariance matrix.

Installation
~~~~~~~~~~~~
.. code-block:: bash

    $ pip install covar

Dependencies
~~~~~~~~~~~~
1. Python (2.7, or 3.3+)
2. numpy
3. scipy (0.16+)
4. cython


.. toctree::
   :maxdepth: 1

.. raw:: html

   <div style="display:none">

.. autosummary::
    :toctree: generated/

    ~cov_shrink

.. raw:: html

   </div>
