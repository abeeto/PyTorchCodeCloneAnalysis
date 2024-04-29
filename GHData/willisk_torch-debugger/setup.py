from setuptools import find_packages, setup

setup(
    name='torchmonitor',
    packages=find_packages(),
    version='0.1.0',
    description='a lightweight torch debugging tool that monitors nans and infs',
    author='Kurt Willis',
    license='BSD',
)
