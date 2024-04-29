#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name='Torcherist',
    version='0.1.2',
    description='utils codes for pyTorch',
    author="Jiashu Xu",
    author_email='1999J0615une@gmail.com',
    url='https://github.com/cnut1648/torcher',
    download_url='https://github.com/cnut1648/torcherist/archive/v0.1.2.tar.gz',
    packages=find_packages(),
    license="MIT license",
    keywords=['nlp', 'skills', 'onet', 'pytorch'],
    install_requires=[
        'beautifulsoup4',
        'numpy',
        'graphviz',
        'matplotlib',
        'requests',
        'nltk',
        'torch',
        'cached_property'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
)
