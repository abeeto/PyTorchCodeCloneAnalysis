import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torte",
    version="0.0.1",
    author="Jacob Moss",
    author_email="cob.mossy@gmail.com",
    description="A small Torch Trainer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mossjacob/torch-trainer",
    project_urls={
        "Bug Tracker": "https://github.com/mossjacob/torch-trainer/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "torte"},
    packages=setuptools.find_packages(where="torte"),
    python_requires=">=3.6",
    install_requires=[
        'torch>=1.7.1',
    ],
)
