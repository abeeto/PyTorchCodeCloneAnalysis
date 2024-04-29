import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torch-helper",
    version="0.5.1-alpha",
    author="Santanu Bhattacharjee",
    author_email="mail.santanu94@gmail.com",
    description="A pytorch library to provide some common utility functions.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/santanu94/torch-helper.git",
    keywords = ['PyTorch', 'Deep Learning'],
    packages=['pthelper'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "torch>=1.0.0",
        "sklearn"
    ]
)
