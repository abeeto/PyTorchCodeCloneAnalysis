from setuptools import setup, find_packages

setup(
    name='slowfast',
    description="SlowFast Video Understanding",
    version='0.1.0',
    zip_safe=False,
    python_requires='>=3.6',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'sf_runner=slowfast.runner:main',
        ],
    },
)
