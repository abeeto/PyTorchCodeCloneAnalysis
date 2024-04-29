import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torch-reproducible-block",        # This is the name of the package
    version="0.0.1",                        # The initial release version
    author="Jerome Abdelnour",                     # Full name of the author
    description="Control random number generator state via reproducible blocks",
    long_description=long_description,      # Long description read from the the readme file
    long_description_content_type="text/markdown",
    url="https://github.com/j3rome/torch-reproducible-block",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],                                      # Information to filter the project on PyPi website
    python_requires='>=3.6',                # Minimum version requirement of the package
    py_modules=["reproducible_block"],             # Name of the python package
    install_requires=[
        'numpy',
        'torch'
    ]
)