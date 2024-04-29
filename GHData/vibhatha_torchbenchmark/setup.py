# Reference: https://github.com/audreyr/cookiecutter-pypackage
# from distutils.core import setup
from setuptools import setup

setup(
    name='torchbenchmark',
    packages=['torchbenchmark', 'torchbenchmark.estimator', 'torchbenchmark.network', 'torchbenchmark.network.core'],
    version='0.0.7',
    description='Pytorch Benchmark Suite',
    author='Vibhatha Abeykoon',
    author_email='vibhatha@gmail.com',
    url='https://github.com/vibhatha/torchbenchmark',
    keywords=['pytorch', 'benchmark', 'gpu', 'performance'],
    python_requires='>=3.6',
    classifiers=[
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Software Development',
    ],
    install_requires=[
        'click',
        'termtables',
        'numpy',
    ],

)
