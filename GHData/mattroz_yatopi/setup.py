#!/usr/bin/env python
from setuptools import find_packages, setup

__version__ = "0.0.1"

if __name__ == '__main__':
    setup(
        name='yatopi',
        version=__version__,
        description='Yet Another Torch Pipeline',
        url='https://github.com/mattroz/yatopi',
        author='Matt Rozanov',
        author_email='matveyrozanov@gmail.com',
        keywords='deep learning, pipeline, pytorch',
        packages=find_packages(exclude=('config', 'tool', 'workspace')),
        license='MIT License',
        zip_safe=False)
