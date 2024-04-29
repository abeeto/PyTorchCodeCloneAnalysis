from setuptools import setup, find_packages

setup(
    name='torch-classifier',
    version='0.0.0',
    author='ryoherisson',
    description='Pytorch implementation for classification',
    url='https://github.com/ryoherisson/torch-classifier',
    packages = find_packages(exclude=("configs", "tests*")),
    python_requires="==3.6.9",
)