# Copyright Hugh Perkins 2015, 2016 hughperkins at gmail
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

from __future__ import print_function
import os
import os.path
import sys
import datetime
import platform
from setuptools import setup
from setuptools import Extension

torch_install_dir = os.getenv('TORCH_INSTALL')
if torch_install_dir is None:
    print('Please set environment variable TORCH_INSTALL to the directory of your torch/install directory')
    sys.exit(1)

osfamily = platform.uname()[0]
print('torch_install:', torch_install_dir)
print('os family', osfamily)

jinja2_only = 'JINJA2_ONLY' in os.environ
cythonize = 'CYTHON' in os.environ

if cythonize:
    from Cython.Build import cythonize

if jinja2_only:
    types = [
        {'Real': 'Long', 'real': 'long'},
        {'Real': 'Float', 'real': 'float'},
        {'Real': 'Double', 'real': 'double'},
        {'Real': 'Byte', 'real': 'unsigned char'}
    ]
    print('Running jinja2...')
    import jinja2
    # generate cython pyx from jinja template...
    from jinja2 import Environment
    env = Environment(loader=jinja2.FileSystemLoader('.'))
    templateNames = [
        'src/PyTorch.jinja2.pyx', 'src/Storage.jinja2.pyx', 'src/PyTorch.jinja2.pxd', 'src/nnWrapper.jinja2.cpp', 'src/nnWrapper.jinja2.h',
        'test/jinja2.test_pytorch.py', 'src/Storage.jinja2.pxd', 'src/nnWrapper.jinja2.pxd', 'src/lua.jinja2.pxd', 'src/lua.jinja2.pyx']
    for templateName in templateNames:
        template = env.get_template(templateName)
        pyx = template.render(
            header='GENERATED FILE, do not edit by hand, ' +
            'Source: ' + templateName,
            header1='GENERATED FILE, do not edit by hand',
            header2='Source: ' + templateName,
            types=types)
        outFilename = templateName.replace('.jinja2', '').replace('jinja2.', '')
        isUpdate = True
        if os.path.isfile(outFilename):
            # read existing file, see if anything changed
            f = open(outFilename, 'rb')  # binary, so get linux line endings, even on Windows
            pyx_current = f.read().decode('utf-8')
            f.close()
            if pyx_current == pyx:
                isUpdate = False
        if isUpdate:
            print(outFilename + ' (changed)')
            f = open(outFilename, 'wb')
            f.write(pyx.encode('utf-8'))
            f.close()
    print('jinja2 finished, exiting')
    sys.exit(0)

compile_options = []
if osfamily == 'Windows':
    compile_options.append('/EHsc')

if osfamily in ['Linux', 'Darwin']:
    compile_options.append('-std=c++0x')
    compile_options.append('-Wno-unused-function')
    compile_options.append('-Wno-unreachable-code')
    compile_options.append('-Wno-strict-prototypes')
    if 'DEBUG' in os.environ:
        compile_options.append('-O0')
        compile_options.append('-g')

runtime_library_dirs = []
libraries = []
extra_link_args = []
# libraries.append('lua5.1')
# libraries.append('luaT')
# libraries.append('mylib')
libraries.append('PyTorchNative')
# libraries.append('TH')
library_dirs = []
# library_dirs.append('cbuild')
library_dirs.append(torch_install_dir + '/lib')

if osfamily != 'Windows':
    runtime_library_dirs = [torch_install_dir + '/lib']

if osfamily == 'Windows':
    libraries.append('winmm')

if osfamily == 'Darwin':  # Mac OS X
    extra_link_args.append('-Wl,-rpath,' + torch_install_dir + '/lib')


def get_file_datetime(filepath):
    t = os.path.getmtime(filepath)
    return datetime.datetime.fromtimestamp(t)

cython_sources = ['src/lua.pyx', 'src/Storage.pyx', "src/PyTorch.pyx"]
ext_modules = []
for cython_source in cython_sources:
    cythoned_filepath = cython_source.replace('.pyx', '.cpp')
    basename = os.path.basename(cython_source).replace('.pyx', '')
    source_name = cythoned_filepath
    if cythonize and (not os.path.isfile(cythoned_filepath) or get_file_datetime(cythoned_filepath) < get_file_datetime(cython_source)):
        source_name = cython_source
    ext_modules.append(
        Extension(basename,
                  sources=[source_name],
                  include_dirs=[torch_install_dir + '/include/TH', 'thirdparty/lua-5.1.5/src', torch_install_dir + '/include'],
                  library_dirs=library_dirs,
                  libraries=libraries,
                  extra_compile_args=compile_options,
                  extra_link_args=extra_link_args,
                  runtime_library_dirs=runtime_library_dirs,
                  language="c++")
    )

if cythonize:
    ext_modules = cythonize(ext_modules)

setup(
    name='PyTorch',
    version='4.1.1-SNAPSHOT',
    author='Hugh Perkins',
    author_email='hughperkins@gmail.com',
    description=(
        'Python wrappers for torch and nn'),
    license='BSD2',
    url='https://github.com/hughperkins/pytorch',
    long_description='Python wrappers for torch and nn',
    classifiers=[
    ],
    install_requires=['numpy'],
    scripts=[],
    ext_modules=ext_modules,
    py_modules=['floattensor', 'PyTorchAug', 'PyTorchHelpers', 'PyTorchLua'],
    package_dir={'': 'src'}
)
