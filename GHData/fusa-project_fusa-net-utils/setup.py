from setuptools import setup, find_packages
import pathlib
here = pathlib.Path(__file__).parent.resolve()

setup(
    name='fusanet_utils',
    version='0.0.1',
    description='Functions to parse datasets and compute features for FUSA models',
    long_description=(here / 'README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    url='https://github.com/fusa-project/fusa-net-utils',
    author='Pablo Huijse',
    author_email='phuijse@inf.uach.cl',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
    keywords='FUSA, neural network, features',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.6, <4',
    install_requires=['numpy', 'pandas', 'torch', 'torchaudio', 'scikit-learn', 'pydub', 'soundfile', 'torchlibrosa', 'tqdm', 'colorednoise', 'torchsampler'],
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/fusa-project/fusa-net-utils',
        'Source': 'https://github.com/fusa-project/fusa-net-utils',
    },
)
