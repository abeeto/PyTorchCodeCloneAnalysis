#!/usr/bin/env python

from setuptools import setup
from setuptools import find_packages


setup(
    name="torch_ignition",
    version="0.0.1",
    url="https://github.com/iKrishneel/torch_ignition.git",
    packages=find_packages(),
    zip_safe=False,
    test_suite="tests",
)
