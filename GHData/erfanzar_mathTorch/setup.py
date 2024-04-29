from setuptools import setup

if __name__ == "__main__":
    setup(name='mathTorch',
          version='0.0.8',
          description='some research for the ones who love the see the deep ends',
          author='erfan zare chavoshi',
          url='https://github.com/erfanzar/',
          author_email='erfanzare82@yahoo.com',
          license='MIT',
          packages=['mathTorch'],
          requires=['numpy', 'numba', 'json5', 'PyYAML'],
          zip_safe=False)
