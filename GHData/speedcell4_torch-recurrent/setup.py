from setuptools import setup

with open('README.md', mode='r', encoding='utf-8') as fd:
    long_description = fd.read()

setup(
    name='torch-recurrent',
    version='0.1.2',
    packages=['torch_recurrent'],
    install_requires=['torch'],
    url='https://github.com/speedcell4/torch-recurrent',
    license='MPL2',
    author='speedcell4',
    author_email='speedcell4@gmail.com',
    description='enhanced recurrent neural networks with PyTorch',
    long_description=long_description,
)
