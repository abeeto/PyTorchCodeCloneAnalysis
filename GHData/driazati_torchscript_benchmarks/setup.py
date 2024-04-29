from setuptools import setup

setup(
    name='torchscript_benchmarks',
    version='1.0',
    description='Utilities for torchscript benchmarks',
    url='http://github.com/driazati/torchscript_benchmarks',
    author='driazati',
    license='GPLv3',
    packages=['torchscript_benchmarks'],
    zip_safe=False,

    # PyTorch is required
    install_requires=['torch'],
)
