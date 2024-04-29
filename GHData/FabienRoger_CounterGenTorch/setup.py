import pathlib
from setuptools import setup, find_packages

"""
To upload the package, run
python setup.py sdist
twine upload --repository-url https://upload.pypi.org/legacy/ dist/countergentorch-VERSION.tar.gz
"""

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()
VERSION = (HERE / "VERSION").read_text()
REQUIREMENTS = (HERE / "requirements.txt").read_text().splitlines()

setup(
    name="countergentorch",
    version=VERSION,
    description="Package providing pytorch model evaluators compatible with countergen, and editing capabilities.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="SaferAI",
    author_email="saferai.audit@gmail.com",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/FabienRoger/CounterGenTorch",
    python_requires=">=3.7",
    include_package_data=True,
    install_requires=REQUIREMENTS,
)
