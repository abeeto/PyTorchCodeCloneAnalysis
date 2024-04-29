import setuptools

with open("readme.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torch-model",
    version="0.0.1",
    author="Igor Smirnov",
    author_email="smirnoviv@rambler.ru",
    description="PyTorch model utils",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/i-v-s/torch-model",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy'
    ],
    python_requires='>=3.6',
)
