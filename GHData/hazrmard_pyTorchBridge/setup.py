from distutils.core import setup



setup(name='pytorchbridge',
      version='0.1.3',
      packages=['pytorchbridge'],
      install_requires=['tqdm', 'scikit-learn>=0.20'],
      author='Ibrahim Ahmed',
      author_email='ibrahim.ahmed@vanderbilt.edu',
      description='Scikit-learn Estimator API for pyTorch Modules',
      url='https://git.isis.vanderbilt.edu/ahmedi/pyTorchBridge',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7'
      ]
)