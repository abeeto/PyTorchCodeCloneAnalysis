import pandas as pd
import matplotlib.pyplot as plt

from cv2 import cv2
from pandas import set_option


def _show_image(dataset: pd.DataFrame, width: int, height: int, index: int) -> None:
    # Picking the image with the corresponding index from the dataset
    image = cv2.imread(fr'./Dataset/Cars License Plates/{dataset["Image_name"].iloc[index]}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize=(width, height))

    x_top = int(dataset["X_Top"].iloc[index] * width)
    y_top = int(dataset["Y_Top"].iloc[index] * height)
    x_bottom = int(dataset["X_Bottom"].iloc[index] * width)
    y_bottom = int(dataset["Y_Bottom"].iloc[index] * height)

    # Adding a rectangle at the corresponding points according to the dataset
    image = cv2.rectangle(image, (x_top, y_top), (x_bottom, y_bottom), (0, 0, 255), 1)
    plt.imshow(image)
    plt.show()


def exploring(dataset: pd.DataFrame, width: int, height: int) -> None:
    print(f'\nData types are: {dataset.dtypes}')
    set_option('display.width', int(1e2))
    set_option('precision', 2)
    print(f'Dataset shape: {dataset.shape}')
    print(f'Description of the dataset:\n{dataset.describe()}')

    # Building a function for visualization
    # visualize_samples: np.ndarray = np.random.randint(0, len(dataset), 2)
    # dataset = dataset.drop(visualize_samples, axis=0)

    # Displaying the 9th car image (in the dataset) with it's highlighted license plate
    _show_image(dataset, width, height, 9)
