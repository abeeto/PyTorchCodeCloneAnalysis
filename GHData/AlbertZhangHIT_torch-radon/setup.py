
import glob
import os

import torch
from setuptools import find_packages
from setuptools import setup
#from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
#from torch.utils.cpp_extension import CUDAExtension

requirements = ["torch", "torchvision"]

def get_extensions():
	this_dir = os.path.dirname(os.path.abspath(__file__))
	extensions_dir = os.path.join(this_dir, "torchradon", "csrc")
	main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
	source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))

	sources = main_file + source_cpu
	extension = CppExtension

	extra_compile_args = {"cxx": []}
	define_macros = []

	sources = [os.path.join(extensions_dir, s) for s in sources]
	include_dirs = [extensions_dir]

	ext_modules = [
		extension(
			"torchradon._C",
			sources,
			include_dirs=include_dirs,
			define_macros=define_macros,
			extra_compile_args=extra_compile_args,
			)
		]
	return ext_modules

setup(
	name="torchradon",
	version="0.2",
	author="hao zhang",
	url="https://github.com/AlbertZhangHIT/torch-radon",
	description="radon transform in pytorch",
	packages=find_packages(exclude=("tests",)),
	ext_modules=get_extensions(),
	cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)