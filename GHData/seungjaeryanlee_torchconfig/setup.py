import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchconfig",  # Replace with your own username
    version="0.1.3",
    author="Seungjae Ryan Lee",
    author_email="seungjaeryanlee@github.com",
    description="TorchConfig is a Python package that simplifies configuring PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seungjaeryanlee/torchconfig",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
