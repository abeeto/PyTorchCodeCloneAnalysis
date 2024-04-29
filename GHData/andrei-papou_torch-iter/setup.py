import typing as t

import setuptools


def read_requirements(path: str) -> t.List[str]:
    with open(path) as f:
        return [line.strip() for line in f]


with open('README.md', 'r') as f:
    long_description = f.read()


setuptools.setup(
    name='torch-iter',
    version='0.0.1',
    description='A collection of PyTorch samplers.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/andrei-papou/torch-iter',
    packages=setuptools.find_packages(exclude=('tests')),
    install_requires=read_requirements('requirements/base/core.txt'),
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Operating System :: OS Independent',
    ],
)
