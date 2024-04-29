from setuptools import setup, find_packages


setup(
    name="tops",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "cloudpickle",
        "omegaconf",
        "easydict",
        "validators"
    ],
)
