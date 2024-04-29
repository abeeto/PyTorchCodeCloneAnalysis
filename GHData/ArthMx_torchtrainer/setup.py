import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='torchtrainer',
    version='0.0.3',
    author='Arthur Moraux',
    author_email='arthur.moraux@gmail.com',
    description='A package for training pytorch models.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/ArthMx/torchtrainer',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
