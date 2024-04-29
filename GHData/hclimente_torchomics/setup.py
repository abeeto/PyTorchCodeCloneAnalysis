from setuptools import find_packages, setup

with open("README.md", "r") as file:
    readme = file.read()

setup(
    name="torchomics",
    packages=find_packages(),
    package_dir={"torchomics": "torchomics"},
    version="0.1.2",
    description="Datasets, transforms and models specific to genomics",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Héctor Climente-González, Joseph C Boyd",
    author_email="hector.climente@riken.jp",
    license="GPL-3",
    url="https://github.com/hclimente/torchomics",
    keywords=["machine-learning", "genomics", "deep-learning"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
    ],
    install_requires=["torch >= 1.12.1", "numpy >= 1.21.5"],
)
