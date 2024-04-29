from setuptools import setup, find_packages
from codecs import open
from os import path

with open('requirements.txt') as f:
    requirements = f.readlines()

setup(
    name='TINX',
    version='0.0.1',
    description='Vanilla PyTorch Testing',
    long_description='A package to test ',
    long_description_content_type='text/markdown',
    url='https://github.com/kanishk16/TINX',
    author='Kanishk',
    author_email='kanishkkalra10@gmail.com',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.6,<3.10',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements
)