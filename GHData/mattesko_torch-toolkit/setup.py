from distutils.core import setup

setup(
    name='torch-toolkit',
    version='0.1',
    packages=['torch-toolkit'],
    license='LICENSE',
    long_description=open('README.md').read(),
    install_requires=[
        "torch >= 1.0",
        "numpy >= 1.17",
        "pydicom >= 2.0",
        "torchvision >= 0.6",
    ],
)