from pathlib import Path
from setuptools import find_packages, setup


ROOT = Path(__file__).parent


setup(
    name='torchx',
    version='1.0.0',
    author='Hui Kang',
    url='https://gitee.com/randydl/torchx',
    description='Utils for PyTorch Training',
    long_description=ROOT.joinpath('README.md').read_text(),
    long_description_content_type='text/markdown',
    python_requires='>=3.7.0',
    install_requires=ROOT.joinpath('requirements.txt').read_text().splitlines(),
    packages=find_packages(),
    include_package_data=True
)
