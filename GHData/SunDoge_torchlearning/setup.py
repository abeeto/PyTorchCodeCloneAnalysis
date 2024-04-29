from setuptools import setup, find_packages

setup(
    name="torchlearning",
    version="0.0.1",
    description="A deeplearning utility library for Pytorch",
    url="https://github.com/chenyaofo/torchlearning",
    author="chenyaofo",
    author_email="chenyaofo@gmail.com",
    packages=find_packages(exclude=['test']),
    install_requires=[
        'torch',
        'py3nvml',
        'Flask',
        'numpy',
        'pyhocon',
        'torchvision',
        'psutil',
        'dataclasses',
        'matplotlib',
        'scipy',
        'Pillow',
    ],
)
