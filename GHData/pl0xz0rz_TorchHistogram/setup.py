# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:58:01 2020

@author: majoi
"""

import setuptools

setuptools.setup(
        name="torchhistogramdd",
        packages=setuptools.find_packages(where='src'),
        package_dir={"": "src"},
        version="0.1",
        install_requires=['torchsearchsorted','torch']
        )