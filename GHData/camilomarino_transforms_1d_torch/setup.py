from setuptools import find_packages, setup

setup(
    name="transforms_1d_torch",
    packages=find_packages(),
    version="0.1.2",
    description="",
    author="Camilo Mariño",
    license="MIT",
    install_requires=[
        "torch",
        "torchvision",  # To use Compose. It should be removed in the future.
    ],
)
