#!/usr/bin/env python

"""The setup script."""
from setuptools import find_packages, setup

with open("README.md", encoding="utf8") as readme_file:
    readme = readme_file.read()


def process_requirements(s: str):
    reqs = []
    for row in s.split("\n"):
        row = row.strip()
        row.replace("==", ">=")  # relaxing the requirements
        if not row.startswith("#") and len(row) > 0:
            reqs.append(row)
    return reqs


with open("./requirements.txt") as r:
    requirements = process_requirements(r.read())

with open("./requirements.dev.txt") as rd:
    requirements_dev = process_requirements(rd.read())

setup(
    author="Gregoire Clement",
    python_requires=">=3.9.*",
    author_email="gc@visium.ch",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.9",
    ],
    description="Unsupervised parts inspection and defect detection",
    install_requires=requirements,
    extras_require={"h5torch-dev": requirements_dev},
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords=["hdf5", "h5py", "torch", "pytorch", "data", "dataloader"],
    name="h5torch",
    packages=find_packages(include=["h5torch", "h5torch.*"]),
    # setup_requires=setup_requirements,
    # test_suite="tests",
    # tests_require=test_requirements,
    url="",
    version="0.0.1",
    zip_safe=True,
    package_data={},
)
