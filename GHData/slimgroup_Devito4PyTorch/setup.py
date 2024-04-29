import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

reqs = ['torch', "devito @ git+https://github.com/devitocodes/devito@master",
        "torchvision"]
setuptools.setup(
    name="devito4pytorch",
    version="0.0.5",
    author="Ali Siahkoohi",
    author_email="alisk@gatech.edu",
    description="Integrating Devito into PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/slimgroup/Devito4PyTorch",
    license='MIT',
    install_requires=reqs,
    packages=setuptools.find_packages()
)
