from setuptools import setup, find_packages
from setuptools.command.install import install
from subprocess import check_call

cuda_deps = ['cupy-cuda112',
	'torch@https://download.pytorch.org/whl/cu111/torch-1.8.1%2Bcu111-cp38-cp38-linux_x86_64.whl',
'torchvision@https://download.pytorch.org/whl/cu111/torchvision-0.9.1%2Bcu111-cp38-cp38-linux_x86_64.whl',
'torchaudio@https://download.pytorch.org/whl/torchaudio-0.8.1-cp38-cp38-linux_x86_64.whl']

cuda_win_deps = ['cupy-cuda112',
	'torch@https://download.pytorch.org/whl/cu111/torch-1.8.1%2Bcu111-cp38-cp38-win_amd64.whl',
'torchvision@https://download.pytorch.org/whl/cu111/torchvision-0.9.1%2Bcu111-cp38-cp38-win_amd64.whl',
'torchaudio@https://download.pytorch.org/whl/torchaudio-0.8.1-cp38-cp38-win_amd64.whl']


setup(
    name='oyDL',
    version='0.0.1',
    description='some torch utilities, Oyler-Yaniv lab @HMS',
    author='Alon Oyler-Yaniv',
    url='https://github.com/oylab/oyDL',
    packages=find_packages(include=['oyDL', 'oyDL.*']),
    python_requires='>=3.8, <3.9',
    dependency_links=[
        "https://download.pytorch.org/whl/torch_stable.html",
    ],
    install_requires=[
        'PyYAML',
	'PyQt5',
    'opencv-python',
	'cloudpickle==1.6.0',
	'dill>=0.3.4',
	'ipython>=7.27.0',
	'ipywidgets>=7.6.3',
	'matplotlib>=3.3.4',
	'napari==0.4.11',
	'numba>=0.53.1',
	'numpy>=1.20.1',
	'pandas>=1.2.4',
	'scikit_image<0.19',
	'scikit_learn>=0.24.2',
	'scipy>=1.6.2',
	'setuptools>=52.0.0',
	'multiprocess>=0.70',
	'jupyter>=1.0.0',
    'torchmetrics',
    'mlxtend',
    'umap-learn'
    ],
	extras_require = {'cuda': cuda_deps,
	'cuda-win': cuda_win_deps,
    },
)
