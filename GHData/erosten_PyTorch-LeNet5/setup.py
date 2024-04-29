from distutils.core import setup
from setuptools import find_packages

requires = [
    'six>=1.10.0',
]



if __name__ == "__main__":
    setup(
        name="PyTorch-LeNet5-util",
        version="0.0.1",
        packages=find_packages('src'),
        package_dir={'': 'src'},
        author='Erik Rosten',
        author_email='erikrrosten@gmail.com',
        install_requires=requires,
        description='',
        include_package_data=True,
    )
