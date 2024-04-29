import setuptools

from torch_fuze.version import version

setuptools.setup(
    name="torch_fuze",
    version=version,
    author="Daiver",
    author_email="ra22341@ya.ru",
    description="Yet another PyTorch training framework",
    long_description="",
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
)