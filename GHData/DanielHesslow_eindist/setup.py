from setuptools import setup

setup(name='eindist',
      version='0.1',
      description='Differentiable einops-style wrapper over torch.distributed.',
      url='https://github.com/DanielHesslow/eindist',
      author='Daniel Hesslow',
      license='MIT',
      packages=['eindist'],
      install_requires=[
          'einops',
          'torch'
      ],
      zip_safe=False)