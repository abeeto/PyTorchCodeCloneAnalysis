from setuptools import setup
import os
import glob
from torch.utils.cpp_extension import BuildExtension, CppExtension

# 头文件目录
include_dirs = os.path.dirname(os.path.abspath(__file__))
# 源代码目录
source_cpu = glob.glob(os.path.join(include_dirs, "cpu", "*.cpp"))

setup(
    name="test_cpp",    # 模块名称，后续在Python中调用
    version="0.1",
    ext_modules=[
        CppExtension("test_cpp", sources=source_cpu, include_dirs=[include_dirs]),
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
