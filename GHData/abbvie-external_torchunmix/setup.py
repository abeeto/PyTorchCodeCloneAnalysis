from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="torchunmix",
    version="0.1",
    description="Automatic stain unmixing and augmentation for histopathology whole slide images in PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abbvie-external/torchunmix",
    author="Erik Hagendorn",
    author_email="erik.hagendorn@abbvie.com",
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Computer Vision",
        "License :: OSI Approved :: MIT License",
    ],
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=['matplotlib', 'tqdm'],
    # List additional groups of dependencies here (e.g. development
    # dependencies). Users will be able to install these using the "extras"
    # syntax, for example:
    #
    #   $ pip install sampleproject[dev]
    #
    # Similar to `install_requires` above, these must be valid existing
    # projects.
    # extras_require={  # Optional
    #     "dev": ["check-manifest"],
    #     "test": ["coverage"],
    # },
    project_urls={  # Optional
        "Bug Reports": "https://github.com/abbvie-external/torchunmix/issues",
        "Source": "https://github.com/abbvie-external/torchunmix/",
    },
)