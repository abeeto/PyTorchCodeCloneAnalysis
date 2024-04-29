import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="k_torch",
    version="0.0.8",
    author="Osama dar, Muhammad Ismail",
    author_email="osamadar1996@gmail.com",
    description="A Keras-like wrapper for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/daroodar/k_torch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)