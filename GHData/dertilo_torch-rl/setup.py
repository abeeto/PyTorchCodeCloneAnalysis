from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    install_requires = f.readlines()

setup(
    name='torch-rl',
    version='0.1',
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    license='LICENSE.txt',
    long_description=open('README.md').read(),
    install_requires=install_requires,
    # dependency_links=['git+https://github.com/maximecb/gym-minigrid.git#egg=gym_minigrid-1.0']#TODO: why is this not working?
)