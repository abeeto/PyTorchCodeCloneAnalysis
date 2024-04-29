try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

long_desc = """
PyTorch modules that make creating flow models much easier.
"""

setup(
    name="torchflow",
    version="0.0.3",
    packages=["torchflow"],
    url="https://github.com/PiotrDabkowski/torchflow",
    install_requires=["torch", "numpy"],
    license="MIT",
    author="Piotr Dabkowski",
    description="PyTorch modules for Normalizing Flows.",
    long_description=long_desc,
)
