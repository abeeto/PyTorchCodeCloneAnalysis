import os
import torch

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
README = open(os.path.join(SCRIPT_DIR, "README.md")).read()


ext_modules = []
if torch.cuda.is_available():
    ext_modules.append(CUDAExtension('_torch_unified_gpu', [
        'torch_unified_gpu.cpp',
    ]))

setup(
    name='torch_unified',
    packages=['torch_unified'],
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    },
    version="0.1",
    description="add support for unified CPU/CUDA memory to pytorch.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/siemanko/torch-unified",
    author="Szymon Sidor",
    author_email="szymon.szymon@gmail.com",
    license="Public Domain",
    install_requires=["torch"],
)
