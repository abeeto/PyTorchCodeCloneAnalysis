from setuptools import setup
from setuptools import find_packages

setup(
    name='torchsampler',
    version='1.0',
    description='up- or down- sampling for imbalanced datsaet.',
    url='https://github.com/HonoMi/pytorch-imbalanced-dataset-sampler',
    author='Honoka',
    install_requires=[
        'torch'
    ],
    packages=find_packages(),
    zip_safe=False,
)
