from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open("torche/version.py") as f:
    exec(f.read())

extra_setuptools_args = dict(
    tests_require=['pytest']
)

setup(
    name ='torche',
    version = __version__,
    author ='Stan Biryukov',
    author_email ='stan0625@uw.com',
    url = 'git@github.com:stanbiryukov/torche.git',
    install_requires = requirements,
    package_data = {'torche':['resources/*']},
    packages = find_packages(exclude=['torche/tests']),
    license = 'MIT',
    description='Torche: Highly performant regularized auto PyTorch feed forward neural network with scikit-learn API and MLFlow tracking.',
    long_description= "Torche is a scikit-learn friendly PyTorch automl solution that provides easy and highly performant results, tracked with MLFlow, for your tabular ML problems.",
    keywords = ['statistics','classification','regression', 'mlflow', 'analysis', 'automl', 'pytorch', 'machine-learning', 'scikit-learn'],
    classifiers = [
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License"
    ],
    **extra_setuptools_args
)