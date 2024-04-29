import os
import re
from setuptools import setup, find_packages


base_dir = os.path.dirname(os.path.abspath(__file__))



def readme():
    with open('README.md') as f:
        return f.read()


def requirements():
    with open('requirements.txt', 'r') as f:
        return f.read().splitlines()


setup(
    name='torch_soft',
    version='0.1.1',
    author='mallikarjun sajjan',
    author_email='flyingmuttus1995@gmail.com',
    description='A high-level deep learning library build on top of PyTorch for classification problems...',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/millermuttu/torch_soft',
    packages=find_packages(),
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    include_package_data=True,
    install_requires=requirements()
)
