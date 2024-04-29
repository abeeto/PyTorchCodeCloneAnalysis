import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torch-segment",
    version="0.0.2",
    author="Sovit Ranjan Rath",
    author_email="sovitrath5@gmail.com",
    description="A library for deep learning image segmentation using PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sovit-123/Torch-Segment",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)