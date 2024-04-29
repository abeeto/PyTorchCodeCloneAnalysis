# -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages
from torch.utils import cpp_extension

##########
# helpers
##########


# Placeholder to change the default
# compilation for command ``build_ext --inplace``.
def get_cmd_classes():
    return {'build_ext': cpp_extension.BuildExtension}


def get_extensions():
    cpp_piecewise_linear = cpp_extension.CppExtension(
        'td3a_cpp_deep.fcts.piecewise_linear_c',
        ['td3a_cpp_deep/fcts/piecewise_linear_c.cpp'])
    ext_modules = [cpp_piecewise_linear]
    return ext_modules


######################
# beginning of setup
######################


here = os.path.dirname(__file__)
if here == "":
    here = '.'
packages = find_packages(where=here)
package_dir = {k: os.path.join(here, k.replace(".", "/")) for k in packages}
package_data = {
    "td3a_cpp_deep.fcts": ['*.cpp', '*.h'],
}

try:
    with open(os.path.join(here, "requirements.txt"), "r") as f:
        requirements = f.read().strip(' \n\r\t').split('\n')
except FileNotFoundError:
    requirements = []
if len(requirements) == 0 or requirements == ['']:
    requirements = []

try:
    with open(os.path.join(here, "readme.rst"), "r", encoding='utf-8') as f:
        long_description = "td3a_cpp:" + f.read().split('td3a_cpp_deep:')[1]
except FileNotFoundError:
    long_description = ""

version_str = '0.0.1'
with open(os.path.join(here, 'td3a_cpp_deep/__init__.py'), "r") as f:
    line = [_ for _ in [_.strip("\r\n ")
                        for _ in f.readlines()]
            if _.startswith("__version__")]
    if len(line) > 0:
        version_str = line[0].split('=')[1].strip('" ')

ext_modules = get_extensions()


setup(name='td3a_cpp_deep',
      version=version_str,
      description="Example of a python module including cython and openmp",
      long_description=long_description,
      author='Xavier Dupré',
      author_email='xavier.dupre@gmail.com',
      url='https://github.com/sdpython/td3a_cpp_deep',
      ext_modules=ext_modules,
      packages=packages,
      package_dir=package_dir,
      package_data=package_data,
      setup_requires=["pybind11", "numpy", "scipy", "torch"],
      install_requires=["pybind11", "numpy", "scipy", "torch"],
      cmdclass=get_cmd_classes())
