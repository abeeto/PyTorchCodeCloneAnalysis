from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import sysconfig

#extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
gone_compile_args = ["-DB32", "-fopenmp"]
#extra_compile_args += ['-DB32']

setup(name='dcgan',
      ext_modules=[CppExtension('dcgan', ['dcgan.cpp'], extra_compile_args = gone_compile_args)],
      include_package_data=True,
      
      include_dirs = ['/home/datalab/graph-one/dist-graph/src', '/home/datalab/graph-one/dist-graph/gview','/home/datalab/graph-one/dist-graph/onedata','/home/datalab/graph-one/dist-graph/dist'],

      cmdclass={'build_ext': BuildExtension},
    
     ## Extension('type', ['type.c'],include_dirs = ['/home/datalab/graph-one/dist-graph/src']),
    )
