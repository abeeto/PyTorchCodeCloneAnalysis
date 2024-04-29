import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torch-tensor-eq",
    version="0.0.1",
    author="Jordan Schneider",
    author_email="jordan.jack.schneider@gmail.com",
    description="Replaces torch's equality with one that returns a bool.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jordan-schneider/torch-tensor-eq",
    packages=setuptools.find_packages(),
)
