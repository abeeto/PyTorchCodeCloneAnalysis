import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="seq2seq-pytorch",  # Replace with your own package name
    version="0.0.1",
    author="Linjian Li",
    author_email="author@example.com",
    description="Python package sequence-to-sequence model with PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LinjianLi/Seq2Seq-PyTorch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        # "python>=3.9",
        "torch>=1.9",
        "matplotlib",
        "prettytable",
        "tqdm",
    ]
)
