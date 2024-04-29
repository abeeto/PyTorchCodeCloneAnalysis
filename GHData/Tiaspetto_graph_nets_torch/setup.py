Skip to content
Search or jump to…
Pull requests
Issues
Marketplace
Explore
 
@Tiaspetto 
deepmind
/
graph_nets
Public
Code
Issues
4
Pull requests
1
Actions
Projects
Security
Insights
graph_nets/setup.py /
@alvarosg
alvarosg Bump version to 1.1.1.dev
Latest commit e7a4bf3 on 29 Jan 2020
 History
 2 contributors
@alvarosg@pbattaglia
63 lines (59 sloc)  2.23 KB

# Copyright 2018 The GraphNets Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Setuptools installation script."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import find_packages
from setuptools import setup

description = """Graph Nets is DeepMind's library for building graph networks in
Tensorflow and Sonnet.
"""

setup(
    name="graph_nets_torch",
    version="0.0.1",
    description="Library for building graph networks in Pytorch.",
    long_description=description,
    author="WarwickAI",
    license="Apache License, Version 2.0",
    keywords=["graph networks", "torch", "machine learning"],
    url="https://github.com/deepmind/graph-nets",
    packages=find_packages(),
    # Additional "tensorflow" and "tensorflow_probability" requirements should
    # be installed separately (See README).
    install_requires=[
        "absl-py",
        "torch"
        "future",
        "networkx",
        "numpy",
        "setuptools",
        "six",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
