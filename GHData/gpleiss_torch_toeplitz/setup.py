import os
import sys
from setuptools import setup, find_packages

import build

this_file = os.path.dirname(__file__)

setup(
    name='torchtoeplitz',
    version='0.1',
    description='Operations for Toeplitz matricies in PyTorch',
    url='https://github.com/gpleiss/torchtoeplitz/',
    author='Geoff Pleiss, Jake Gardner',
    author_email='geoff@cs.cornell.edu, jrg365@cornell.edu',
    # Require cffi.
    install_requires=['cffi>=1.4.0'],
    setup_requires=['cffi>=1.4.0'],
    # Exclude the build files.
    packages=find_packages(exclude=['build']),
    # Package where to put the extensions. Has to be a prefix of build.py.
    ext_package='',
    # Extensions to compile.
    cffi_modules=[
        os.path.join(this_file, 'build.py:ffi')
    ],
)
