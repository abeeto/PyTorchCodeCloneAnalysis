import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchvision_utils",
    version="0.0.2",
    author="pejmans21",
    author_email="pezhmansamadi21@gmail.com",
    description="some function for pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pejmanS21/torchvision_utils",
    project_urls={
        "Bug Tracker": "https://github.com/pejmanS21/torchvision_utils/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "lib"},
    packages=setuptools.find_packages(where="lib"),
    python_requires=">=3.6",
)