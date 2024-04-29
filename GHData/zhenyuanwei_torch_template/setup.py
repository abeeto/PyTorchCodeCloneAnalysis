from setuptools import setup, find_packages

maintainer = 'Kevin, Wei'
maintainer_email = 'kevin@yxtechs.cn'
author = maintainer
author_email = maintainer_email
description = "'This is a front for pytorch lib in deep learning"
version = '0.01'
name = 'torch template'

long_description = """
Tobe added from readme
"""

install_requires = [
    'numpy>=1.11.1',
    'scipy>=0.18.0',
]

packages = [
    'torch_template',
]
platforms = ['linux', 'macOS']
url = 'https://github.com/zhenyuanwei/torch_template'
download_url = ''
classifiers = [
    'Development Status :: 3 - Alpha',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
#   'Programming Language :: Python :: 2',
#   'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
]

setup(author=author,
      author_email=author_email,
      description=description,
      license=license,
      long_description=long_description,
      install_requires=install_requires,
      maintainer=maintainer,
      name=name,
      packages=find_packages(),
      platforms=platforms,
      url=url,
      download_url=download_url,
      version=version,
      classifiers=classifiers)