from setuptools import setup
import re
import os
import sys


setup(
    name="flexinet",
    version="0.0.4",
    python_requires=">3.6.0",
    author="Michael E. Vinyard - Harvard University - Massachussetts General Hospital - Broad Institute of MIT and Harvard",
    author_email="mvinyard@broadinstitute.org",
    url="https://github.com/mvinyard/flexinet",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    description="Flexible torch neural network architecture API",
    packages=[
        "flexinet",
        "flexinet._models",
        "flexinet._models._supporting_functions",
        "flexinet._io",
        "flexinet._preprocessing",
        "flexinet._utilities",
    ],
    
    install_requires=[
        "anndata>=0.7.8",
        "numpy>=1.17.0",
        "torch>=1.10.1",
        "licorice_font>=0.0.3",
        "geomloss>=0.2.3",
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.6",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    license="MIT",
)
