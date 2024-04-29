from setuptools import setup
import glob
import os


def find_stubs(package):
    BASE_PATH = os.path.dirname(__file__)
    CWD = os.getcwd()
    try:
        os.chdir(os.path.join(BASE_PATH, package))
        return {package: glob.glob("**/*.pyi", recursive=True)}
    finally:
        os.chdir(CWD)


setup(
    name="torchvision-stubs",
    maintainer="Federico Sallemi",
    maintainer_email="federico.sallemi@contentwise.tv",
    description="PEP 561 type stubs for torchvision",
    url="https://pytorch.org/docs/master/torchvision/",
    version="2019.6.0",
    packages=["torchvision-stubs"],
    install_requires=["torchvision>=0.3.0"],
    package_data=find_stubs("torchvision-stubs"),
)
