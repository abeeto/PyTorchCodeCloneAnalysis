from setuptools import setup

setup(
    name='torchkit',
    version='0.0.1',
    author='Tivadar Danka',
    author_email='85a5187a@opayq.com',
    description='A modular active learning framework for Python3',
    license='MIT',
    packages=['torchkit', 'torchkit.models', 'torchkit.tools'],
    classifiers=['Development Status :: 4 - Beta'],
    install_requires=['scikit-image', 'numpy', 'pillow', 'six', 'torchvision'],
)