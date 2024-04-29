import os
import shutil
import subprocess
import distutils.command.clean
from pathlib import Path
from setuptools import find_packages, setup

ROOT_DIR = Path(__file__).parent.resolve()


def _get_version():
    version = '0.1.0a0'
    sha = 'Unknown'
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT_DIR)).decode('ascii').strip()
    except Exception:
        pass

    if os.getenv('BUILD_VERSION'):
        version = os.getenv('BUILD_VERSION')
    elif sha != 'Unknown':
        version += '+' + sha[:7]

    return version, sha


def _export_version(version, sha):
    version_path = ROOT_DIR / 'axolotls' / 'version.py'
    with open(version_path, 'w') as f:
        f.write("__version__ = '{}'\n".format(version))
        f.write("git_version = {}\n".format(repr(sha)))


VERSION, SHA = _get_version()
_export_version(VERSION, SHA)

print('-- Building version ' + VERSION)

pytorch_package_version = os.getenv('PYTORCH_VERSION')

pytorch_package_dep = 'torch'
if pytorch_package_version is not None:
    pytorch_package_dep += "==" + pytorch_package_version


class clean(distutils.command.clean.clean):
    def run(self):

        # Run default behavior first
        distutils.command.clean.clean.run(self)

        # Remove axolotls extension
        for path in (ROOT_DIR / 'axolotls').glob('**/*.so'):
            print(f'removing \'{path}\'')
            path.unlink()
        # Remove build directory
        build_dirs = [
            ROOT_DIR / 'build',
        ]
        for path in build_dirs:
            if path.exists():
                print(f'removing \'{path}\' (and everything under it)')
                shutil.rmtree(str(path), ignore_errors=True)


setup(
    # Metadata
    name='axolotls',
    version=VERSION,
    description='Lightweight DataFrame library on PyTorch',
    url='https://github.com/wenleix/axolotls',
    author='Wenlei Xie',
    author_email='wenlei.oss.pure@gmail.com',
    license='BSD',
    install_requires=[
        'requests',
        'tabulate',
        pytorch_package_dep
    ],
    python_requires='>=3.7',
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    # Package Info
    packages=find_packages(exclude=["test*", "benchmark*", "demo*"]),
    zip_safe=False,
    cmdclass={
        'clean': clean,
    },
)