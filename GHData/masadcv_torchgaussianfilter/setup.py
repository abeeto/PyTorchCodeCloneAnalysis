import setuptools

def readme():
    with open('README.md') as f:
        return f.read()

setuptools.setup(name='torchgaussianfilter',
      version='0.0.2',
      description='Gaussian filtering using PyTorch',
      long_description=readme(),
      long_description_content_type="text/markdown",
      classifiers=[
        'Operating System :: OS Independent',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
      ],
      keywords='gaussian filtering 2d 3d medical features hand-crafted',
      url='http://github.com/masadcv/torchgaussianfilter',
      author='Muhammad Asad',
      author_email='masadcv@gmail.com',
      license='BSD-3-Clause',
      packages=['torchgaussianfilter'],
      install_requires=[
          'torch',
      ],
      zip_safe=False)
