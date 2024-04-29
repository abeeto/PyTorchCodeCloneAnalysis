import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchsweeper-crnbaker",
    version="0.0.1",
    author="Christian Baker",
    author_email="christian.baker@kcl.ac.uk",
    description="Classes for timing and parameter sweeping PyTorch and SciPy code.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/crnbaker/torchsweeper",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
       'numpy>=1',
       'torch>=1.7'
    ],
    python_requires='>=3.6'
)
