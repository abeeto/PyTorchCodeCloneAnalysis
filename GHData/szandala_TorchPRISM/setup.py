import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchprism",
    version="2.0.0",
    author="Tomasz Szandala",
    author_email="tomasz.szandala@gmail.com",
    description="Principal Image Sections Mapping for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/szandala/TorchPRISM",
    packages=["torchprism"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
        install_requires=[
        "torch >= 1.1",
        "torchvision >= 0.3.0"
    ],
    keywords=["deep-learning", "PCA", "visualization", "interpretability"]
)

# rm -rf build/ dist/ torchprism.egg-info/
# python setup.py sdist bdist_wheel
# twine upload dist/*
