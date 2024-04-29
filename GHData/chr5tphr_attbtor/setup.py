#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name = "attbtor",
    version = "0.1",
    packages=find_packages(),
    install_requires=[
        'ntorx',
        'numpy>=1.15.4',
        'Pillow>=5.3.0',
        'torch>=1.0.1',
        'torchvision>=0.3.0',
        'Click>=7.0',
    ],
    entry_points={
        'console_scripts': [
            'attbtor = attbtor.cli:main'
        ]
    }
)
