from setuptools import setup, find_packages

version = "0.0.1"

with open("torchtest/__init__.py", "w") as fh:
    fh.write(f"__version__ = '{version}'\n")

setup(
    name="torchtest_shaliulab",
    version=version,
    packages = find_packages(),
    extras_require={
    },
    install_requires=[
        "torch", "numpy"
    ],
    entry_points={
        "console_scripts": [
            "torchtest=torchtest.torchtest:main",
            ]
    },
)



