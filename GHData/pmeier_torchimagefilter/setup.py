from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = ("numpy", "torch")

testing_requires = (
    "pyimagetest@https://github.com/pmeier/pyimagetest/archive/master.zip",
    "pillow",
    "torchvision",
)

classifiers = (
    "Development Status :: 3 - Alpha",
    "License :: OSI Approved :: BSD License",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering",
)

setup(
    name="torchimagefilter",
    description="Image filtering in PyTorch",
    version="0.2-dev",
    url="https://github.com/pmeier/torchimagefilter",
    license="BSD-3",
    author="Philip Meier",
    author_email="github.pmeier@posteo.de",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("test",)),
    install_requires=install_requires,
    extras_require={"testing": testing_requires,},
    python_requires=">=3.6",
    classifiers=classifiers,
)
