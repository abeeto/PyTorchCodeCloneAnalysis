import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchlazy", # Replace with your own username
    version="0.0.1",
    author="Ankit Kumar",
    author_email="kitofans@gmail.com",
    description="Lazy module creation for pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/an-kumar/torchlazy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
