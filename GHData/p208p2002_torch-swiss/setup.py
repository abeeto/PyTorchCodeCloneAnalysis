from setuptools import setup, setuptools
import os

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

try:
    lastest_git_tag = list(os.popen("git tag"))[0].split()[-1]
except:
    lastest_git_tag = 'latest'

setup(
    name="torch_swiss",
    version=lastest_git_tag,
    author='Philip Huang',
    author_email="p208p2002@gmail.com",
    description="Toolkit for PyTorch like a Swiss Knife",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/p208p2002/pretty-print-confusion-matrix",
    packages=setuptools.find_packages(),
    install_requires=[
       'sklearn',
       'torch',
    ],
    python_requires='>=3.5',
)