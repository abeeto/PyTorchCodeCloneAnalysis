try:
    from setuptools import setup
except:
    from distutils.core import setup

setup(
    name='pytorch_custom',
    version='0.0dev',
    author='Alexander Soare',
    packages=['pytorch_custom'],
    url='https://github.com/alexander-soare/PyTorch-Custom',
    license='Apache 2.0',
    description='My own miscellaneous helpers for pytorch',
    install_requires=[
        'pandas',
        'matplotlib',
        'tqdm',
        'numpy',
        'scikit-learn',
    ],
)