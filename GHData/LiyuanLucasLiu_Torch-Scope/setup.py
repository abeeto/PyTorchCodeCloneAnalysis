#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

def read_readme():
    with open('README.md') as f:
        return f.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'numpy',
    'tensorboardX',
    'gitpython',
    'torch',
    'gspread',
    'PyOpenSSL',
    'oauth2client'
]

setup(
    name='torch-scope',
    version='0.5.5',
    description='A Toolkit for Training, Tracking and Saving PyTorch Models',
    long_description= read_readme(),
    long_description_content_type="text/markdown",
    author='Lucas Liu',
    author_email='llychinalz@gmail.com',
    url='https://github.com/LiyuanLucasLiu/Torch-Scope',
    packages=find_packages(exclude=['docs']),
    include_package_data=True,
    install_requires=requirements,
    license='Apache License 2.0',
    entry_points={
        'console_scripts': ['torch_scope=torch_scope.commands:run'],
    },
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',        
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ]
)

# python setup.py sdist bdist_wheel --universal
# twine upload dist/*