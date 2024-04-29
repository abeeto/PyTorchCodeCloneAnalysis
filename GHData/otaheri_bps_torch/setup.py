from setuptools import setup, find_packages


setup(
    name='bps_torch',
    include_package_data=True,
    #packages=find_packages(),
    description='A Pytorch Inplementation of bps_torch 3D representation',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    version='0.1',
    url='https://github.com/otaheri/bps_torch',
    author='Omid Taheri',
    author_email='omid.taheri@tuebingen.mpg.de',
    maintainer='Omid Taheri',
    maintainer_email='omid.taheri@tuebingen.mpg.de',
    #keywords=['pip','MANO'],
    install_requires=[
          'numpy>=1.16.2',
          'torch>=1.0.1.post2',
          'torchgeometry>=0.1.2',
      ],
    packages=['bps_torch']
      
    )
