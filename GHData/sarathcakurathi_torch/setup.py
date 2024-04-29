import re
from setuptools import setup, find_packages

version = re.search('^__version__\s*=\s*"(.*)"', open('torch/torch.py').read(), re.M).group(1)

with open("README.md", "rb") as f:
    long_descr = f.read().decode("utf-8")

setup(
    name = "torch-cli",
    packages = find_packages(include=['torch', 'torch.*']),
    entry_points = {
        "console_scripts": ['torch = torch.torch:main']
        },
    package_data={
        "": ["*.conf"],
    },
    version = version,
    description = "Python command line tool for Terraform Orchestration",
    long_description = long_descr,
    author = "Sarath C Akurathi",
    author_email = "sarath.c.akurathi@gmail.com",
    url = "https://github.com/sarathcakurathi/torch",
    )