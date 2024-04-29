#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools


with open('README.md', 'r', encoding='utf-8') as f:
    README = f.read()

setuptools.setup(
    name='PyTorch-Bayes',
    version='0.0.3',
    description='A simple PyTorch wrapper making Bayesian learning much easier',
    author='Yi Zhang',
    author_email='yizhang.dev@gmail.com',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/imyizhang/PyTorch-Bayes',
    download_url='https://github.com/imyizhang/PyTorch-Bayes',
    packages=setuptools.find_packages(),
    keywords=[
        'pytorch', 'bayesian'
    ],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
    license='MIT',
    python_requires='>=3.8',
    install_requires=[
        'torch'
    ],
)
