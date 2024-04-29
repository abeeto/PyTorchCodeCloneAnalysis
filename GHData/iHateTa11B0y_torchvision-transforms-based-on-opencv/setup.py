from setuptools import setup, find_packages
import sys, os

install_requires = [
        'torch',
        'torchvision',
        'numpy',
        'opencv-python',
        ]

long_description = '''
This repository is intended to offer some common augmentataion functions in computer vision task base on opencv. The functions' prototype comes from FAIR's maskrcnn-benchmark. I tried my best to implement these functions strictly follow the details of torchvision and pillow. Any discussions are welcomed.
'''

setup(
    name='cvtorch',
    version='0.0.9',
    description='vision tools based on opencv',
    long_description=long_description,
    author='iHateTa11B0y',
    author_email='1187203155@qq.com',
    install_requires=install_requires,
    packages=["cvtorch"],

)
