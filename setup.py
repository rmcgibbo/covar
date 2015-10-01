"""Shrinkage covariance estimator for python
"""
from setuptools import setup, Extension
from Cython.Distutils import build_ext


##########################
VERSION = "0.1"
__version__ = VERSION
##########################

DOCLINES = __doc__.split("\n")
CLASSIFIERS = """\
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: BSD
Programming Language :: Python
Operating System :: OS Independent
"""

extensions = [
    Extension('covar', ['covar.pyx'], language='c++'),
]

setup(
    name='covar',
    author="Robert T. McGibbon",
    author_email='rmcgibbo@gmail.com',
    cmdclass={'build_ext': build_ext},
    url="https://github.com/rmcgibbo/covar",
    description=DOCLINES[0],
    install_requires=['scipy >= 0.16', 'numpy >= 0.16'],
    long_description="\n".join(DOCLINES[2:]),
    version=__version__,
    license='BSD',
    zip_safe=False,
    ext_modules=extensions,
)
