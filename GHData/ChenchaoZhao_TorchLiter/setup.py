from setuptools import find_packages, setup

from src.torchliter import __version__

# load readme
with open("README.md", "r") as f:
    long_description = f.read()


setup(
    name="torchliter",
    version=__version__,
    author="Chenchao Zhao",
    author_email="chenchao.zhao@gmail.com",
    description="A lightweight training tool for pytorch projects.",
    package_dir={"": "src"},
    packages=find_packages("src", exclude=["tests"]),
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["torch", "numpy", "dataclasses"],
    license="MIT",
)
