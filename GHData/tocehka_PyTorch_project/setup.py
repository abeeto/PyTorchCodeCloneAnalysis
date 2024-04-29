from setuptools import setup, find_packages

setup(
    name="Machine search and DB merge",
    version="0.0.1",
    packages=find_packages(),
    include_package_data=True,
    url="https://github.com/tocehka/PyTorch_project",
    author="tocehka",
    install_requires=[
        "hnswlib>=0.3.4",
        "bs4>=0.0.1",
        "numpy>=1.18.2",
        "fse>=0.1.15",
        "gensim>=3.8.3",
        "torch>=1.5.0",
        "nltk>=3.5",
        "requests>=2.23.0"
    ]
)
