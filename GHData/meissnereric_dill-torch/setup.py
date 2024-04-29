from setuptools import setup, find_packages
import re

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='dilltorch',  # this is the name of the package as you will import it i.e import package-name
    version=0.1,
    author='Eric Meissner',
    author_email='meissner.eric.7@gmail.com',
    description='Deep Linear Neural Networks (DLNN)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/meissnereric/dill-torch',
    packages=find_packages(exclude=['tst*']),
    include_package_data=True,
    license='Apache License 2.0',
    classifiers=(
       'Development Status :: 1 - Planning',
       'Intended Audience :: Developers',
       'Intended Audience :: Education',
       'Intended Audience :: Science/Research',
       'Programming Language :: Python :: 3',
       'Programming Language :: Python :: 3.4',
       'Programming Language :: Python :: 3.6',
       'Programming Language :: Python :: 3.7',
       'License :: OSI Approved :: Apache Software License',
       'Operating System :: OS Independent',
       'Topic :: Scientific/Engineering :: Artificial Intelligence',
       'Topic :: Scientific/Engineering :: Mathematics',
       'Topic :: Software Development :: Libraries',
       'Topic :: Software Development :: Libraries :: Python Modules'
    ),
)
