import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="trop", # Replace with your own username
    version="0.0.1",
    author="Anand Srinivasan",
    author_email="anand@a0s.co",
    description="Transfer Operators in PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ooblahman/trop",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)