import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchfetch",
    version="0.0.1",
    author="Jaemin Son",
    author_email="woalsdnd@gmail.com",
    description="Fetch pytorch data and models without pain.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jaeminSon/torchfetch",
    project_urls={
        "Bug Tracker": "https://github.com/jaeminSon/torchfetch/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=[
        'pandas',
        'numpy',
        'albumentations',
        'opencv_python_headless',
        'opencv-python',
        'torchvision',
        'pycocotools',
        'Pillow'
    ],
    extras_require={
    ':python_version == "3.6"': [
        'torch==1.7.0',
    ],
    ':python_version == "3.7"': [
        'torch==1.7.0',
    ],
    },
)
