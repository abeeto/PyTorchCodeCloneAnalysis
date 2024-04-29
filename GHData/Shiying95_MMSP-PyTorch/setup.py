from setuptools import setup, find_packages

setup(
    name="mmsp",
    packages=find_packages(
        exclude=[".dfc", ".vscode", "dataset", "notebooks", "result", "scripts"]
    ),
    version="1.0.0",
    license="MIT",
    description="MMSP: Multi-Modal Sale Prediction",
    author="Shiying Ni",
    author_email="nisy13@163.com",
    url="https://github.com/Shiying95/MMSP-PyTorch",
    keywords=["multi-modal sale prediction"],
    install_requires=["torch", "pytorch_lightning"],
)
