import config

from AI_Model import build_AI_model
from data_exploring import exploring
from loading_data import data_loading
from data_preprocessing import _pre_processing
from kaggle_data_loading import kaggle_data_loading


def license_plate_detection():
    WIDTH = int(config.width)  # Recommended value of 224
    HEIGHT = int(config.height)  # Recommended value of 224
    CHANNEL = int(config.channels)

    # Loading the Dataset from Kaggle, creating a DataFrame from, and saving it into a .cvs file for later use
    # Deprecated. The Dataset on Kaggle was blocked by DataTurks!
    # URL: https://www.kaggle.com/dataturks/vehicle-number-plate-detection
    # kaggle_data_loading()

    # Pre processing the dataset images and it's corresponding .csv labels (including data filtering and cleaning)
    # FIXME: Important Note
    # User needs to call the below function (pre_processing) at least once when running the code.
    # The below data_loading() function assumes that some files have been created and saved, and will try to load them.
    # The problem is that one specific file - Dataset/images.npy wasn't added to Git due to large size (> 100MB).
    # So, run it once, for the mentioned file to be created and after, one can put it in commit and not use it!
    # _pre_processing()

    # Loading the data
    dataset, mean, std = data_loading()

    # Exploring the data
    exploring(dataset, WIDTH, HEIGHT)

    # Build, Train, Validate, and Test a Neural Network AI Model
    build_AI_model(dataset, CHANNEL, WIDTH, HEIGHT)
