"""
1、方式一
从github上安装
pip install https://github.com/wucng/torchTools/archive/master.zip

2、方式二
在setup.py目录下执行以下命令安装
python setup.py install

3、方式三
python setup.py build
python setup.py install

4、方式四
1.在setup.py目录下执行
python  setup.py  sdist
# 最终生成一个dist文件夹，在文件夹里面就有一个创建好的安装包，
# 格式为xxx.tar.gz的压缩包
2.安装xxx.tar.gz包
pip install xxx.tar.gz
3.检查是否安装成功
pip list   # 显示所有已安装的包
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*_
# from distutils.core import setup
from setuptools import setup, find_packages

# pip freeze >requirements.txt
requirements = [
    "torch>=0.4.0+",
    "torchvision",
    'numpy', 'pandas',
    'matplotlib', 'pillow', 'tqdm', 'sklearn',
    'opencv-python','opencv-contrib-python',
    'imageio','scipy','scikit-image','six','imgaug'
]

setup(
    name='torchTools',  # 包的名字
    version='0.0.1',  # 版本号
    description='some general tools for Classification ,Object detection '
                'and Object segmentation GAN and so on',  # 描述
    author='wucng',  # 作者
    author_email='goodtensorflow@gmail.com',  # 你的邮箱**
    url='https://github.com/wucng/',  # 可以写github上的地址，或者其他地址
    download_url='',
    packages = find_packages(),  # 包含所有torchTools中的包

    # 依赖包
    install_requires=requirements,
    license="MIT",
    zip_safe=False,
    keywords=['torch',"computer vision","deep learning"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ]
)
