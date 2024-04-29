from distutils.core import setup

setup(
    name='colin-crying-detector',
    version='0.0.1',
    data_files = [('data/models', ['data/models/stage-2.pth'])],
    scripts=['bin/detect.py',],
)
