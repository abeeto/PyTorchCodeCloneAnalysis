"""Author: Brandon Trabucco, Copyright 2019"""


from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = [
    'torch',
    'tensorboardX',
    'numpy',
    'matplotlib',
    'gym',]


setup(
    name='torchforce', version='0.1',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=[p for p in find_packages() if p.startswith('torchforce')],
    description='A fast and efficient framework for training hierarchical RL models in pytorch.')