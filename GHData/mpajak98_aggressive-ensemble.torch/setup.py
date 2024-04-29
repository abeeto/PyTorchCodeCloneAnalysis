import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="aggressive_ensemble",
    version="0.4.0",
    author="Maciej Pajak",
    author_email="mpajak98@gmail.com",
    description="A package implementing aggressive ensemble methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mpajak98/aggressive-ensemble.pytorch",
    project_urls={
        "Documentation": "https://github.com/mpajak98/aggressive-ensemble.pytorch/docs",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6, <9",
    install_requires=[
        "pandas",
        "torch",
        "torchvision",
        "matplotlib",
        "numpy",
        "imgaug>=0.4.0",
        "Pillow",
        "scikit-learn",
        "setuptools",
        "tqdm"],
)