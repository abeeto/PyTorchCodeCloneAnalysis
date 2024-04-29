from setuptools import setup, find_packages


setup(
   name='torch_helpers',
   version='0.0.4',
   description='Utility tools for PyTorch',
   author='Felix Abrahamsson',
   author_email='FelixAbrahamsson@github.com',
   keywords='pytorch torch helpers utils',
   packages=['torch_helpers'],
   install_requires=[
       'torch',
   ],
)
