
# make sure torch is installed
import os
os.system('python3 -m pip install torch')

from setuptools import setup
setup(
    name='torch-rusty1s-loader',
    version='0.1',
    install_requires=[
        "torch-bincount",
        "torch-cluster",
        "torch-geometric",
        "torch-scatter",
        "torch-sparse",
        "torch-spline-conv"
    ]
)
