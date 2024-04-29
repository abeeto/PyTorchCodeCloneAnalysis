from pathlib import Path
from setuptools import setup, find_packages
setup(
    name=Path(__file__).parent.name,
    version="0.0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
