import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name="torch_runner",
    packages=find_packages(),
    version="0.1.0",
    license="MIT",
    description="Trainer for Pytorch",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Rohith Gandhi G",
    author_email="grohith327@gmail.com",
    url="https://github.com/grohith327",
    include_package_data=True,
    install_requires=["torch", "tqdm"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)