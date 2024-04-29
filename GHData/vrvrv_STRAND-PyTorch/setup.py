from setuptools import setup, find_packages

setup(
    name='strand',
    version='0.0',
    url='https://github.com/vrvrv/STRAND-PyTorch',
    description='Python implementation of STRAND',
    install_requires=[
        'numpy',
        'torch==1.9.0',
        'pytorch-lightning==1.5.1',
        'scikit-learn==1.0.1',
        'scipy==1.7.1',
        'rich',
        'wandb',
        'hydra-core==1.1.0',
        'hydra-colorlog==1.1.0',
        'hydra-optuna-sweeper==1.1.0',
    ],
    packages=find_packages('src')
)
