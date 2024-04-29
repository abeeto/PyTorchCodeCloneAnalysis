#!/usr/bin/env python

from setuptools import setup, find_packages


with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='mnistclassifier',
    version='0.0.1',
    description='classifier for tasks like mnist in torch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/chriamue/mnistclassifier',
    author='Christian',
    license='BSD',

    packages=find_packages(exclude=['docs', 'tests']),
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.5',
    install_requires=[
        'scipy',
        'scikit-learn',
        'pandas'
    ],
    extras_require={
        'dev': [
            'coverage',
            'factory_boy',
            'IPython>=7.0.1',
            'm2r',
            'mock',
            'pytest>=3.6.3',
            'pytest-flask>=0.13.0',
            'tox',
        ],
        'docs': [
            'sphinx',
            'sphinx-autobuild',
            'sphinx-click',
            'sphinx-rtd-theme',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    entry_points='''
        [console_scripts]
    ''',
)
