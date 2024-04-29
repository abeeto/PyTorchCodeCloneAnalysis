from setuptools import setup, find_packages

setup( 
    name = 'dna-yolov5-torch',
    version = '1.0.1',
    description = 'YOLO v5 wrapper for DNA framework',
    author = 'Kang-Woo Lee',
    author_email = 'kwlee@etri.re.kr',
    url = 'https://github.com/kwlee0220/dna-yolov5-torch',
    install_requires = [
        'opencv-python',
        'pandas',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'requests',
        'pyyaml',
        'tqdm',
    ],
    packages = find_packages(),
    #include_package_data = True,

    python_requires = '>=3',
    zip_safe = False
)