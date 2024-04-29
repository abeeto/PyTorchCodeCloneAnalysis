from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ARGS = [('DIM', '128'), ('NWINDOWS', '64'), ('BLOCKSIZE', '64')]
DEBUG_FLAGS = [] # ['-lineinfo', '-G', '-g']

setup(name='torch-horcorr',
    version='1.0.0',
    description='Horizontal correlation between tow tensors with shape (batch, channels, height, width0) and (batch, channels, height, width1)',
    author='Thomas Pönitz',
    author_email='tasptz@gmail.com',
    package_dir={'': 'py'},
    packages=['horcorr'],
    ext_modules=[
        CUDAExtension('horcorr.horcorr_cuda',
            sources=[
                'src/horcorr_cuda.cpp',
                'src/horcorr_cuda_kernel.cu'
            ],
            define_macros=ARGS,
            extra_compile_args={
                'cxx': [],
                'nvcc': ['-D' + '='.join(a) for a in ARGS] + DEBUG_FLAGS
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires='torch>=1.1',
    setup_requires='torch>=1.1',
    python_requires='>=3.6')
