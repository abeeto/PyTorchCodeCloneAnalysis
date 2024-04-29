from setuptools import find_packages, setup


setup(
    name="svgd-torch",
    version='0.1.0',
    python_requires='>=3',
    packages=find_packages(),
    install_requires=['numpy', 'torch', 'scikit-learn', 'scipy'],
    license='GNU',
)