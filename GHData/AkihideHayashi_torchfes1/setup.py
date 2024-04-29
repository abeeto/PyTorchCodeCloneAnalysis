"""TorchFES."""
from setuptools import find_packages, setup

setup(
    name="torchfes",
    version="0.0.0",
    install_requires=["torch", "h5py"],
    packages=find_packages(),
)
