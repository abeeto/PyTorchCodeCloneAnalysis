import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchreinforce",
    version="0.1.0",
    author="Federico A. Galatolo",
    author_email="galatolo.federico@gmail.com",
    description="A pythonic implementation of the REINFORCE algorithm that is actually fun to use",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/galatolofederico/torchreinforce",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch",
        "numpy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
        "Development Status :: 4 - Beta"
    ],
)