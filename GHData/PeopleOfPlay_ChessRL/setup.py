from setuptools import setup

setup(
    name='ChessRL',
    version='1.0',
    packages=['ChessRL'],
    url='https://github.com/PeopleOfPlay/ChessRL.git',
    license='MIT',
    author='Marc Henriot, Adrien Turchini',
    description='Package for our final project of IFT_7201 at ULaval',
    install_requires=['torch', 'tqdm', 'numpy', 'chess', 'matplotlib', 'poutyne']
)