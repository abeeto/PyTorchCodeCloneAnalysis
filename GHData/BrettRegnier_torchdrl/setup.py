import setuptools
from setuptools import setup

requirements = ['torch>=1.6.0', 'torchvision>=0.7.0', 'numpy']

setup(name='torchdrl',
    version='0.2',
    description='Deep Reinforcement Learning module for torch',
    url='http://github.com/brettregnier/torchdrl',
    author='Brett Regnier',
    author_email='bretternestregnier@gmail.com',
    license='MIT',
    packages=setuptools.find_packages(),
    zip_safe=False,
    install_requires=requirements)