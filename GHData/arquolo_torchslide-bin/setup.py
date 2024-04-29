import sys
from pathlib import Path

import setuptools

if not ((3, 6) <= sys.version_info < (3, 7)):
    raise OSError(f'This module supports only Python 3.6')


class BinaryDistribution(setuptools.Distribution):
    def has_ext_modules(self):
        return True


platform = ''
for key, platform in {'--windows': 'win_amd64',
                      '--linux': 'manylinux1_x86_64'}.items():
    if key in sys.argv[1:]:
        sys.argv.remove(key)
        sys.argv += ['-p', platform]
        break
else:
    if 'bdist' in sys.argv[1:]:
        raise ValueError(f'specify either --windows or --linux')


setuptools.setup(
    name='torchslide',
    version='0.3.0',
    url='https://github.com/arquolo/torchslide-bin',
    author='Paul Maevskikh',
    author_email='arquolo@gmail.com',
    description='torchslide - prebuilt version for Python 3.6',
    long_description=Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    package_data={
        '': ['**/*.dll', '**/*.pyd'] if 'win' in platform else ['**/*.so'],
    },
    python_requires='>=3.6, <3.7',
    install_requires=[
        'dataclasses',
        'numpy>=1.15',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
    ],
    distclass=BinaryDistribution,
)
