from setuptools import setup,find_packages
setup(name="modelzoo",
      version='0.0.0',
      description='model zoo with pytorch.',
      url="https://github.com/jimmy0087/modelzoo-master",
      author='JimmyYoung',
      license='JimmyYoung',
      packages= find_packages(),
      #packages = ['faceai','faceai/Alignment/*.py','faceai/Detection/*.py','faceai/ThrDFace/*.py'],
      zip_safe=False
      )