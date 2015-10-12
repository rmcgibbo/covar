.. currentmodule:: covar

Covar: shrinkage covariance estimation
=======================================

This Python package contains two functions, :func:`cov_shrink_ss` and
:func:`cov_shrink_rblw` which implements plug-in shrinkage estimators for the covariance matrix.

The :func:`cov_shrink_ss` estimator is described by `Schafer and Strimmer (2005) <http://www.degruyter.com/view/j/sagmb.2005.4.1/sagmb.2005.4.1.1175/sagmb.2005.4.1.1175.xml>`_, where it is
called "Target D: (diagonal, unequal variance)". The :func:`cov_shrink_rblw` estimator is described by `Chen  Yilun, Wiesel, and Hero (2009) <http://tbayes.eecs.umich.edu/_media/yilun/covestimation/chen_icassp1_09.pdf>`_.

.. figure:: /../tests/test_2.png
    :width: 600

Installation
~~~~~~~~~~~~
.. code-block:: bash

    $ pip install covar

Dependencies
~~~~~~~~~~~~
Python (2.7, or 3.3+), Numpy (1.6 or later), Scipy (0.16 or later), Cython


.. toctree::
   :maxdepth: 1

.. raw:: html

   <div style="display:none">

.. autosummary::
    :toctree: generated/

    covar.cov_shrink_ss
    covar.cov_shrink_rblw

.. raw:: html

   </div>
