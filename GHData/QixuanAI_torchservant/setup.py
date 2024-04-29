# -*- coding: utf-8 -*-
# @Time    : 2019/09/15 19:34
# @Author  : LQX
# @Email   : qxsoftware@163.com
# @File    : setup.py
# @Software: PyCharm

from setuptools import setup, find_packages

long_description = "TorchServant is an assembly helping quickly development for PyTorch users." + \
                   "It contains following components:\n" + \
                   "\tmodelservant:\tManage weights files and checkpoints, keep training process continuous.\n" + \
                   "\tprocmonitor:\t\tAPI for visdom and tensorboard, visualize diagrams, illustrations and progresses.\n" + \
                   "\tstats\t\tResource statistics on GPUs, memories, cpu, and consumed time.\n" + \
                   "\tclassicmodels:\tInclude AlexNet, VGG, Resnet, Inception, etc.\n" + \
                   "\tvisualfeature:\tVisualize feature maps during training and evaluation process.\n" + \
                   "\tweightransfer:\tAn Qt-based visual tool to transfer weights between different models."

setup(
    name="torchservant",
    version="0.2.2",
    keywords={"pytorch", "tool", "deeplearning", "util", "servant", "visualize"},
    description="Many useful tools for PyTorch.",
    long_description=long_description,
    license="GNU GENERAL PUBLIC LICENSE v3",
    url="https://github.com/QixuanAI/torchservant",
    author="qxsoftware",
    author_email="qxsoftware@163.com",
    packages=find_packages(exclude=["checkpoints", "logs", "test*", ]),
    install_requires=["torch"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0',
)
