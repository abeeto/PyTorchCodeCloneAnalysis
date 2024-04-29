from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="torchplus",
    version="0.1.0",
    packages=find_packages(),
    description="Useful extras when working with pytorch",
    install_requires=requirements,
)
