from setuptools import setup, find_packages

with open("README.rst", "r") as fh:
    long_description = fh.read()

dependencies = [
    "torch~=1.7.0",
    "setuptools~=50.3.2",
    "opencv-python~=4.4.0.44",
    "numpy~=1.19.2",
    "PyYAML~=5.3.1",
    "pandas~=1.1.3",
    "joblib~=0.17.0",
    "scikit-learn~=0.23.2"
]

setup(
    name="torch-cv",
    version="0.1.2",
    author="Raghhuveer Jaikanth",
    author_email="raghhuveerj97@gmail.com",
    description="A high level package for Computer Vision with pytorch",
    long_description=long_description,
    long_description_content_type="text/x-rst ",
    url="https://github.com/RJaikanth/torch-cv",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 4 - Beta",
        "Natural Language :: English",
        "Operating System :: Unix"
    ],
    python_requires=">=3.7",
    scripts=["scripts/tcv"],
    install_requires=dependencies
)
