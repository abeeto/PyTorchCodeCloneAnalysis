import glob
import os
import shutil
import subprocess
import sys

from setuptools import setup, find_packages

install_require_list = [
    "torch>=1.11",
]

doc_require_list = [
    "sphinx",
    "sphinx-rtd-theme",
    "sphinx-gallery",
    "matplotlib"
]

if __name__ == "__main__":
    import setuptools
    import setuptools.command.build_ext
    from setuptools.command.install import install

    setup(
        name="torchoptim",
        version='0.0.1',
        author="Alpa team",
        author_email="",
        description="torchoptim",
        long_description="torchoptim",
        long_description_content_type="text/markdown",
        url="https://github.com/yf225/torchoptim",
        classifiers=[
            'Programming Language :: Python :: 3',
            'Topic :: Scientific/Engineering :: Artificial Intelligence'
        ],
        keywords=(""),
        packages=find_packages(),
        python_requires='>=3.7',
        install_requires=install_require_list,
        extras_require={
            'doc': doc_require_list
        },
    )
