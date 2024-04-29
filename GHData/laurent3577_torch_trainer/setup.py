from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='torch_trainer',
    url='https://github.com/laurent3577/torch_trainer.git',
    author='Laurent Dillard',
    author_email='laurent.dillard@gmail.com',
    packages=setuptools.find_packages(),
    install_requires=['torch','yacs'],
    version='0.1.1',
    license='MIT',
    description='Python package containing handy torch wrappers for DL research',
)