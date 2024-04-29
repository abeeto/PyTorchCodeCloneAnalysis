from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize('scnet3d_preprocess_cython.pyx'))
