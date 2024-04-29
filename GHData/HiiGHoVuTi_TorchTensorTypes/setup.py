import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torch-tensor-type", # Replace with your own username
    version="0.1.0",
    author="Maxime",
    author_email="himyundevacc@gmail.com",
    description="Practical Pipelining for pyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HiiGHoVuTi/TorchTensorTypes",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
