from setuptools import setup, find_packages

setup(name='PyTorch_Template', 
      version='0.0.0',
      author='Matt Purri',
      packages=find_packages(exclude=['experiments']),
      )