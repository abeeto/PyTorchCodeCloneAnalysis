
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='torchtraineretnai',
    version='1.0.2',
    author="Daniele Calanna",
    description="Deep learning trainer for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DanieleCalanna/PyTorchTrainer",
    packages=setuptools.find_packages(exclude=["examples", "images"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "torch",
        "torchvision",
        "torchnet",
        "tqdm",
    ],
)
