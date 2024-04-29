#!/usr/bin/env python
import os
import io
import re
import shutil
import sys
from setuptools import setup, find_packages
from pkg_resources import get_distribution, DistributionNotFound

def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def get_dist(pkgname):
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

VERSION = find_version('blowtorch', '__init__.py')

readme = open('README.md').read()

requirements = [
    'numpy',
    'torch',
    'torchnet',
    'torchvision'
]

setup(
    #meta
    name='blowtorch',
    version=VERSION,
    author='Pratik C.',
    author_email='pratik.ac@gmail.com',
    long_description=readme,
    url='',
    license='LGPL',

    packages=find_packages(exclude=('test',)),
    zip_safe=True,
    install_requirements=requirements,
)