from setuptools import setup, find_packages

__version__ = '1.0.11'
URL = 'https://github.com/tczhangzhi/torcheeg'

install_requires = [
    'tqdm>=4.64.0', 'numpy>=1.21.5', 'pandas>=1.3.5', 'scipy>=1.7.3',
    'scikit-learn>=1.0.2', 'lmdb>=1.3.0', 'einops>=0.4.1', 'mne>=1.0.3',
    'xmltodict>=0.13.0', 'networkx>=2.6.3', 'PyWavelets>=1.3.0',
    'spectrum>=0.8.1', 'torchmetrics>=0.8.2', 'mne_connectivity>=0.4.0'
]

test_requires = ['pytest>=7.1.2']

example_requires = ['pytorch-lightning']

pyg_requires = [
    'torch-scatter',
    'torch-sparse',
    'torch-cluster',
    'torch-spline-conv',
    'torch_geometric>=2.0.3',
]

readme = open('README.rst').read()

setup(
    name='torcheeg',
    version='1.0.11',
    description=
    'TorchEEG is a library built on PyTorch for EEG signal analysis. TorchEEG aims to provide a plug-and-play EEG analysis tool, so that researchers can quickly reproduce EEG analysis work and start new EEG analysis research without paying attention to technical details unrelated to the research focus.',
    license='MIT',
    author='TorchEEG Team',
    author_email='tczhangzhi@gmail.com',
    keywords=['PyTorch', 'EEG'],
    url=URL,
    packages=find_packages(),
    long_description=readme,
    python_requires='>=3.7',
    extras_require={
        'example': example_requires,
        'test': test_requires,
        'pyg': pyg_requires
    },
    install_requires=install_requires)