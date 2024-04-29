from setuptools import setup, find_packages

if __name__ == '__main__':
    setup(
        packages=find_packages(),
        name='tutil',
        author='Maxim Kochurov <maxim.v.kochurov@gmail.com>',
        short_description='Torch utils shared between projects',
        install_requires=[
            'torchvision',
            'numpy'
        ]
    )
