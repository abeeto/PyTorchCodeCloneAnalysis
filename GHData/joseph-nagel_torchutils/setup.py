from setuptools import setup, find_packages

NAME = 'torchutils'
VERSION = '0.0.1'
AUTHOR = 'Joseph Nagel'
EMAIL = 'JosephBNagel@gmail.com'
URL = 'https://github.com/joseph-nagel/torchutils'
LICENSE = 'MIT'
DESCRIPTION = 'Keras-like convenience for PyTorch'

try:
    with open('README.md', 'r') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name = NAME,
    version = VERSION,
    author = AUTHOR,
    author_email = EMAIL,
    url = URL,
    license = LICENSE,
    description = DESCRIPTION,
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    packages = find_packages(),
    install_requires = [
        'numpy',
        'torch',
        'torchvision'
    ],
    python_requires = '>=3.6'
)

