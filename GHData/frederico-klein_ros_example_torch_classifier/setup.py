from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
d = generate_distutils_setup(
    packages=['ros_example_torch_classifier'],
    package_dir={'': 'src'}
)
setup(**d)
