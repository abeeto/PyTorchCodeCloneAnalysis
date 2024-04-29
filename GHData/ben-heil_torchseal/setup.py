import os
from setuptools import setup

# Don't overwrite pytorch version if it's already installed (logic from
# https://github.com/pytorch/vision/blob/master/setup.py)
pytorch_dep = 'torch'
if os.getenv('PYTORCH_VERSION'):
    pytorch_dep += "==" + os.getenv('PYTORCH_VERSION')

requirements = [
    pytorch_dep,
]

setup(name='torchseal',
      version='1.0',
      description='A utility for finding memory leaks in pytorch',
      author='Ben Heil',
      author_email='ben.jer.heil@gmail.com',
      url='https://github.com/ben-heil/torchseal',
      packages=['torchseal'],
      install_requires=['torch'],
      extras_require={
          'tests': ['torch',
                    'torchvision',
                    'pytest'
                    ]
      },
      )
