from setuptools import setup, find_packages

setup(
    name="pcl_torch",
    description="PyTorch implementation for Path Consistency Learning",
    version="0.1",
    python_requires=">=3.10",
    install_requires=[
        "black==22.3.0",
        "numpy==1.23.2",
        "torch==1.12.1",
    ],
    packages=find_packages(),
    include_package_data=True,
)
