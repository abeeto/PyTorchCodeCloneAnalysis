from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bmv',
    ext_modules=[
        CUDAExtension('bmv', [
            'bmv.cpp',
            'bmv_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
