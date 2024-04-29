from setuptools import setup

# Config
seed = 42  # for reproducibility
training_split_ratio = 0.9  # use 90% of samples for training, 10% for testing
num_epochs = 5

# If the following values are False, the models will be downloaded and not computed
compute_histograms = False
train_whole_images = False
train_patches = False

setup(
    name='Torch',
    version='',
    packages=[''],
    url='',
    license='',
    author='LG',
    author_email='',
    description=''
)
