from setuptools import setup
from os import path

with open(path.join(path.abspath(path.dirname(__file__)), 'VERSION'), encoding='utf-8') as f:
    version = f.read()

setup(name='torch_helpers',
      description='A set of helper functions for pyTorch',
      long_description='Torch helpers is a collection of useful helpers and debugging tools for PyTorch. Most notably'
                       'the torch computation graph graphing tool!',
      version=version,
      url='https://github.com/episodeyang/torch_helpers',
      author='Ge Yang',
      author_email='yangge1987@gmail.com',
      license=None,
      keywords=['pyTorch', 'torch', 'deep learning', 'debugging'],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3'
      ],
      packages=['torch_helpers'],
      install_requires=['moleskin']
      )
