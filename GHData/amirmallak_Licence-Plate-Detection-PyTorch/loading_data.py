import config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, Tuple


def data_loading() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    # FIXME: Important Note
    # Couldn't upload to Git the Dataset/images.npy due to large file size (> 100MB)
    images_path = config.images_path
    plates_path = config.plates_path
    images_statistics_path = config.images_statistics_path
    license_plate_dataset_path = config.license_plate_dataset_path

    images: np.ndarray = np.load(images_path, allow_pickle=True)
    images: Dict = images[()]

    plates: np.ndarray = np.load(plates_path, allow_pickle=True)
    plates: Dict = plates[()]

    image_statistics: np.ndarray = np.load(images_statistics_path, allow_pickle=True)
    image_statistics: Dict = image_statistics[()]
    mean, std = image_statistics['mean'], image_statistics['std']

    dataset: pd.DataFrame = pd.read_csv(license_plate_dataset_path)

    # Explore one image
    # Create figure and axes
    figure, axis = plt.subplots(2, 1, constrained_layout=True)

    # Setting titles
    axis[0].set_title('Car Image')
    axis[1].set_title('License Plate Image')

    # Display the images
    axis[0].imshow(images['Car_License_Image_0'])
    axis[1].imshow(plates['Car_License_Image_0'])

    plt.show()

    print(f'Total number of car images in the dataset is: {dataset.shape[0]}\n')

    return dataset, mean, std
