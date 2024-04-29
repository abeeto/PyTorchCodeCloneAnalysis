# change to the directory of this script
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# remove -c from sys.argv

setup(name='pybind_cuda',
      ext_modules=[CppExtension('pybind_torch', ['main.cpp', 'modules/cpp_extension.cpp', 'modules/cuda_extensions.cpp', 'modules/cuda_kernel.cu'])],
      cmdclass={'build_ext': BuildExtension})