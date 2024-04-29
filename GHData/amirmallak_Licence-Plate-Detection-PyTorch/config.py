from os import getenv
from dotenv import load_dotenv

"""
This module is a configuration module. It saves (maps) all the needed default values for certain parameters, and configs
required configurations for connecting to the desired DB.
"""

load_dotenv()  # A function for handling a .env file with the necessary configurations

model_path = getenv('MODEL_PATH', None)
channels = getenv('CHANNEL', None)
width = getenv('WIDTH', None)
height = getenv('HEIGHT', None)
batch_size = getenv('BATCH_SIZE', None)

images_path = getenv('IMAGES_PATH', None)
plates_path = getenv('PLATES_PATH', None)
images_statistics_path = getenv('IMAGES_STATISTICS_PATH', None)
license_plate_dataset_path = getenv('LICENSE_PLATE_DATASET_PATH', None)
data_dir = getenv('DATA_DIRECTORY', None)
