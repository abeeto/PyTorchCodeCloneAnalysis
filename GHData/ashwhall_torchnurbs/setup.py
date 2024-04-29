from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='torchnurbs',
    version='0.1.0',
    description='Differentiable NURBS curve and surface evaluation using PyTorch',
    long_description=readme,
    author='Ash Hall',
    author_email='ashwhall@gmail.com',
    url='https://github.com/ashwhall/torchnurbs',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

