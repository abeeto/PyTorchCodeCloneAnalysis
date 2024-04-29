#!/usr/bin/env python
# -*- coding:utf-8 -*-


from setuptools import setup, find_packages

setup(
    name="torchmods",
    version="0.5.20",
    keywords=("cv", "pytorch", "auxiliary"),
    description="mods for torch & cv",
    long_description="experimental mods for pytorch & cv research",
    license="MIT Licence",

    url="https://github.com/klrc/torchmods",
    author="klrc",
    author_email="sh@mail.ecust.edu.com",

    packages=find_packages(),
    include_package_data=True,
    platforms=["all"],
    install_requires=[
        "torch",
        "matplotlib",
        "paramiko",
        "opencv_python",
        "requests",
        "numpy",
        "torchvision",
        "beautifulsoup4",
        "imageio",
        "Pillow",
        "pynvml",
    ]
)
