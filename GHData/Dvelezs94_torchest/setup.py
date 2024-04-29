from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

# This gets deployed when a new release is made by github actions
VERSION = '{{VERSION_PLACEHOLDER}}'
DESCRIPTION = 'Toolbelt for pytorch framework'
LONG_DESCRIPTION = 'Pytorch tools and utilities (Trainers, data generators, functions, and more...)'

setup(
    name="torchest",
    version=VERSION,
    author="Diego Velez",
    author_email="diegovelezs94@gmail.com",
    description=DESCRIPTION,
    url = "https://github.com/Dvelezs94/torchest",
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['numpy', 'matplotlib', 'tqdm', 'torch', 'wandb'],
    keywords=['python', 'ai', 'machine learning', 'neural network', 'pytorch', 'trainer'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)