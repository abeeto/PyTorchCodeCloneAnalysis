from setuptools import setup, find_packages

setup(
    name="torchlight",
    version="1",
    packages=find_packages(),
    install_requires=["numpy", "scipy", "scikit-learn", 
                      "scikit-image", "pyyaml"],
    author="l3robot"
)