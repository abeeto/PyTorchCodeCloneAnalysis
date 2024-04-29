#! python

from setuptools import setup, find_packages

with open("./description.txt") as f:
    desctiprion = f.read()

setup(
    name="torch_jtnn",
    version="1.0.0",
    author="TakashiKusachi",
    description=desctiprion,
    install_requires=[
        'torchvision',
        'torch',
    ],
    extras_require={
        'example':[
            ],
        'doc':[
            'sphinx'
            ],
    },
    packages=find_packages(exclude=['example','docs']),
    entry_points={
        'console_scripts':[
            'make_vocab = torch_jtnn.scripts:cli_make_vocab',
            'jtnn_prepro = torch_jtnn.scripts:cli_jtnn_prepro',
            'jtnn_train = torch_jtnn.scripts:cli_jtnn_train'
        ]
    }
)