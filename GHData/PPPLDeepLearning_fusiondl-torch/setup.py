from setuptools import setup, find_packages

import plasma.version

# try:
#     os.environ['MPICC'] = subprocess.check_output(
#         "which mpicc", shell=True).decode("utf-8")
# except BaseException:
#     print("Please set up the OpenMPI environment")
#     exit(1)

setup(
    name="plasma",
    version=plasma.version.__version__,
    packages=find_packages(),
    # scripts = [""],
    description="PyTorch models for fusion disruption prediction",
    long_description="""Add description here""",
    author="Ge Dong, Kyle Gerard Felker",
    author_email="gdong@princeton.edu",
    maintainer="Kyle Gerard Felker",
    maintainer_email="felker@anl.gov",
    # url = "http://",
    download_url="https://github.com/PPPLDeepLearning/fusiondl-torch",
    # license = "Apache Software License v2",
    # test_suite="tests",
    install_requires=[
        "pathos",
        "matplotlib>=2.0.2",
        "pre-commit",
        # 'scikit-learn',
        # 'joblib',
    ],
    tests_require=[],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: System :: Distributed Computing",
    ],
    platforms="Any",
)
