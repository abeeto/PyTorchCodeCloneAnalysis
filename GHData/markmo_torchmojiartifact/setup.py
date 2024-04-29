import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='torchmojiartifact',
    version='0.0.4',
    author='Mark Moloney',
    author_email='m4rkmo@gmail.com',
    description='BentoML artifact framework for the Torchmoji Model',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/markmo/torchmojiartifact',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'BentoML==0.10.1',
    ],
    python_requires='>=3.6',
)
