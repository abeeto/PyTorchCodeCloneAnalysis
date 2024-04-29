
from setuptools import setup, find_packages

setup (
    name='dellve_torch_imagenet',
    version='0.0.1',
    packages=find_packages(),
    install_requires=['dellve'],
    entry_points='''
    [dellve.benchmarks]
    AlexnetBenchmark=dellve_torch_imagenet.benchmark:AlexnetBenchmark
    '''
)
