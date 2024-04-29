from setuptools import setup, find_packages

setup(
    name='torchrl',
    packages=[
        package for package in find_packages() if package.startswith('torchrl')
    ],
    version='0.1.0',
)
