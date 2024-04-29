#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name = "ntorx",
    version = "0.1",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.15.4',
        'Pillow>=5.3.0',
        'scikit-image>=0.14.1',
        'torch>=0.4.1.post2',
        'torchvision>=0.2.1',
    ],
)
