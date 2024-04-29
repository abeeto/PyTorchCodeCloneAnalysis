from setuptools import setup, find_packages

requires = [
    'tropofy',
]

setup(
    name='tropofy-bridge-torch',
    version='1.0',
    description='Bridge and Torch Problem',
    author='Tropofy',
    url='http://www.tropofy.com',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requires,
)
