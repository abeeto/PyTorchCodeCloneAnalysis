# Copyright 2022 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from setuptools import setup, find_packages

setup(
    name="torch-sputnik",
    version="0.1.0",
    description="A PyTorch port of the Sputnik library for Sparse Matrix Multiplication Kernels for GPUs",
    long_description="""""",
    url="https://github.com/siddharth9820/sputnik",
    author="Siddharth Singh, Abhinav Bhatele",
    author_email="ssingh37@umd.edu, bhatele@cs.umd.edu",
    classifiers=["Development Status :: 2 - Pre-Alpha"],
    keywords="sputnik, sparse matrix multiplication, pytorch",
    packages=find_packages(),
    install_requires=["torch"],
)
