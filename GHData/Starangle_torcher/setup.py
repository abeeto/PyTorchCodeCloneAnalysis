import setuptools

setuptools.setup(
    name="torcher",
    version="0.0.5",
    author="binsu",
    author_email="binsu.cs@qq.com",
    description="A pytorch model training util",
    url="https://github.com/Starangle/torcher",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)