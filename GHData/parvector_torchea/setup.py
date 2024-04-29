import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchea",
    version="0.1",
    author="Mirzoev Parviz",
    author_email="parvector@yandex.com",
    description="Training and construction of torch models based on evolutionary algorithms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/parvector/torchea",
    project_urls={
        "Bug Tracker": "https://github.com/parvector/torchea/issues",
    },
    packages=setuptools.find_packages(where="."), #
)