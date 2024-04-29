from setuptools import find_packages, setup

setup(
    name='aidapt_torch_utils',
    packages=find_packages(),
    version='0.1.0',
    description='Collection of utils for PyTorch training and models',
    author='AIDAPT',
    license='MIT',
    requires=["torch"]
)