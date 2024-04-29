# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r") as file:
    long_description = file.read()

setuptools.setup(
    name="torchx",
    version="0.0.8",
    author="AntonioJKim",
    author_email="contact@antoniojkim.com",
    description="An 'eXtension' to the PyTorch Deep Learning Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/antoniojkim/TorchX",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    include_package_data=True,
    install_requires=["numpy", "pyyaml", "torch>=1.4"],
)
