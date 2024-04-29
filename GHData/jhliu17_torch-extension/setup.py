from setuptools import setup


setup(
    name='torch_extension',
    version='0.0.1',
    description='a simple case to illustrate how to extend pytorch',
    author='Junhao Liu',
    author_email='junhaoliu17@gmail.com',
    keywords=['pytorch', 'extension'],
    packages=['torch_extension'],
    include_package_data=True
)
