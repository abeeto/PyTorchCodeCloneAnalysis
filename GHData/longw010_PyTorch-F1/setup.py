import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchf1", # Replace with your own username
    version="0.1.0",
    author="Long Wang",
    author_email="lwang010@gmail.com",
    description="todo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/longw010/PyTorch-F1",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
