from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

files = ['src/hello.cpp']
this_dir = [os.path.dirname(os.path.abspath(__file__))]

extra_compile_args = {"cxx": ["-std=c++14"]}
setup(
    name="mymould",
    ext_modules=[
        CppExtension(name="hellworld",
                     sources=files,
                     extra_compile_args=['-wd4624'],
                     libraries=[],
                     include_dirs=this_dir)
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)


