from pathlib import Path

from setuptools import find_packages, setup


class Package:
    name = 'torch-parameter-groups'
    module = name.replace("-", "_")
    description = 'Group PyTorch Parameters according to Rules'
    keywords = 'PyTorch Parameter Groups'
    package_data = {module: ['schema/*.json']}
    entry_points = {}

    here = Path(Path(__file__).parent).absolute()
    with open(here / 'README.md') as f:
        long_description = f.read()

    with open(here / 'requirements.txt') as f:
        install_requires = f.read().splitlines()

    __version__: str = None
    with open(here / module / '__version__.py') as f:
        exec(f.read())


setup(
    name=Package.name,
    version=Package.__version__,
    description=Package.description,
    long_description=Package.long_description,
    long_description_content_type='text/markdown',
    url=f'https://github.com/FebruaryBreeze/{Package.name}',
    license='MIT',
    author='SF-Zhou',
    author_email='sfzhou.scut@gmail.com',
    python_requires='>=3.6.0',
    keywords=Package.keywords,
    packages=find_packages(exclude=['tests']),
    package_data=Package.package_data,
    entry_points=Package.entry_points,
    install_requires=Package.install_requires
)
