#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
  readme = readme_file.read()

requirements = [
  'numpy',
  'SimpleITK==1.2.4',
  'torch>=1.6',
  'torchvision',
  'tqdm',
  'torchio==0.17.50',
  'torchsummary',
]

setup(
  name='TorchIO_Minimal',
  version='0.0.1', # NR: non-release; this should be changed when tagging\
  author="Sarthak Pati",
  author_email='software@cbica.upenn.edu',
  python_requires='>=3.6',
  install_requires=requirements,
  license="BSD-3-Clause License",
  long_description=readme,
  long_description_content_type='text/markdown',
  include_package_data=True,
  zip_safe=False,
)
