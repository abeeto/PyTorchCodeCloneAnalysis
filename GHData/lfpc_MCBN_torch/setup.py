from setuptools import setup, find_packages

setup(
    # Needed to silence warnings (and to be a worthwhile package)

    name='uncertainty_estimation',

    url='https://github.com/lfpc/MCBN_torch',
    author='Luís FP Cattelan',
    author_email='luisfelipe1998@gmail.com',
    # Needed to actually package something
    packages=find_packages(),
    # Needed for dependencies
    install_requires=['numpy','torch'],
    # *strongly* suggested for sharing
    version='1.0',
    # The license can be anything you like
    license='MIT',
    description='MonteCarlo BatchNormalization implementation in pytorch',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
