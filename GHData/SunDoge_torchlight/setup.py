from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

setup(
    name='torchlight',
    version='0.1.0',
    description='Torchlearning Copy',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/SunDoge/torchlight',
    author='SunDoge',
    packages=find_packages(exclude=['contrib', 'docs', 'tests'])
)
