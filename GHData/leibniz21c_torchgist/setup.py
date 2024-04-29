from setuptools import setup, find_packages

setup(
    name='torchgist',
    version='1.0.0',
    description='Pytorch GIST descriptor reimplementation.',
    author='HEESUNG YANG',
    author_email='leibniz21c@gmail.com',
    url='https://github.com/ndo04343/torchgist',
    download_url='https://github.com/ndo04343/torchgist/archive/master.zip',
    install_requires=[
        'torch',
        'numpy',
    ],
    packages=find_packages(exclude=[]),
    keywords=[
        'gist',
        'gist descriptor',
        'gist feature',
        'lmgist',
        'computer vision',
    ],
    python_requires='>=3.6',
    package_data={},
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)