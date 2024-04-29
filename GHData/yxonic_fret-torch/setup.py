#!/usr/bin/env python
from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(
    name='fret-torch',
    version='0.1.0',
    url='https://github.com/yxonic/fret-torch',
    license='MIT',
    author='yxonic',
    author_email='yxonic@gmail.com',
    description='PyTorch bindings for fret.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['fret_torch'],
    platforms='any',
    python_requires='>=3.4',
    install_requires=[
        'fret>=0.3.3',
        'torch>=1.0.0'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ],
)
