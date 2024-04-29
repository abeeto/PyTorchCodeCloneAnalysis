import subprocess
import sys
from setuptools import setup, find_packages

# Hack to get around dependency-links issue.
subprocess.call([sys.executable, "-m", "pip", "install", "torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html"])

requirements = [
    "matplotlib",
    "numpy",
    "pre-commit",
    "black",
	"imageio",
	"imblearn",
	"jupyter",
	"matplotlib",
	"numpy",
	"pandas",
	"progressbar",
	"sklearn",
	"tensorflow-datasets",
	"akida",
	"cnn2snn",
	"akida-models"
]

setup(
    name="Akida Torch",
    version="0.1.0",
    description="Akida API in PyTorch",
    author="Erik Lanning",
    author_email="cs@eriklanning.com",
    packages=find_packages(exclude=("tests", "examples")),
    install_requires=requirements,
    python_requires=">=3.7.0",
)
