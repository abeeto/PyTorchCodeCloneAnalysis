import setuptools


setuptools.setup(
    name='tnet',
    version='0.0.1alpha1',
    url='https://github.com/mhjabreel/tnet',
    license='Apache 2.0',
    install_requires=['theano'],
    author='Mohammed Jabreel',
    author_email='mhjabreel@gmail.com',
    description='Torch and torchnet like library for building and training neural networks in Theano',
    packages=setuptools.find_packages()
)
