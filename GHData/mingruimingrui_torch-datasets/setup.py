import setuptools

setuptools.setup(
    name='torch-datasets',
    version='0.2b',
    description='Dataset tools for quick creation and editting of datasets',
    url='https://github.com/mingruimingrui/torch-datasets',
    author='Wang Ming Rui',
    author_email='mingruimingrui@hotmail.com',
    packages=[
        'torch_datasets',
        'torch_datasets.datasets',
        'torch_datasets.collate_containers',
        'torch_datasets.samplers',
        'torch_datasets.utils'
    ]
)
