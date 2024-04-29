from setuptools import setup, find_packages

setup(
    name='ros_workshop',
    version='0.1.0',
    install_requires=[
        'torch', 
        'imageio',
        'matplotlib',
        'tqdm',
        'scikit-image',
        'numpy',
        'torchvision',
        'visdom'
        ]
)