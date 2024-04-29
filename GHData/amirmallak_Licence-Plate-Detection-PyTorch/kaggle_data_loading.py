import os
import torch
import urllib
import numpy as np
import pandas as pd
import urllib.request

from PIL import Image
from typing import Dict


def kaggle_data_loading():
    path = r'./Dataset/Indian_Number_plates.json'

    # The Json file in Kaggle is not in a json format. Instead, each row is in a json format
    df: pd.DataFrame = pd.read_json(path, lines=True)

    # Creating a Dataset data frame for Neural Network Model
    dataset: Dict = {"Image_name": [],
                     "Image_height": [],
                     "Image_width": [],
                     "X_Bottom": [],
                     "Y_Bottom": [],
                     "X_Top": [],
                     "Y_Top": []}

    # A dictionary that will map an image name (key) to a np.array which represents the Car image (value)
    images: Dict = {}
    # A dictionary that will map an image name (key) to a np.array which represents the Licence Plate image (value)
    plates: Dict = {}

    # Creating a directory to save all the dataset's images in
    if not os.path.isdir(r'./Dataset'):
        os.mkdir(r'./Dataset')
    dir_path = r'./Dataset/Cars License Plates'
    if not os.path.isdir(dir_path):
        os.mkdir(r'./Dataset/Cars License Plates')

    means, stds = [], []
    for index, row in df.iterrows():
        req = urllib.request.Request(row["content"], headers={'User-Agent': 'Mozilla/5.0'})
        webpage = urllib.request.urlopen(req, timeout=10).read()
        # webpage = urllib.request.urlopen(row["content"])
        with Image.open(webpage) as image:
            image = image.convert('RGB')

            # Resizing image. Some with height of 225/224/177 etc. Scaling to a uniform size of 128x128x3
            # image.resize((WIDTH, HEIGHT, CHANNEL))
            image.save(fr'./Dataset/Cars License Plates/Car_License_Image_{index}.jpeg', "JPEG")
            images[f'Car_License_Image_{index}'] = np.array(image)

            image_tensor = torch.Tensor(np.array(image).flatten())
            means.append(torch.mean(image_tensor))
            stds.append(torch.std(image_tensor))

        dataset["Image_name"].append(f'Car_License_Image_{index}.jpeg')

        data_annotation: Dict = row["annotation"][0]

        # Saving the original Height and Width of the image (just for formality)
        dataset["Image_height"].append(data_annotation["imageHeight"])
        dataset["Image_width"].append(data_annotation["imageWidth"])
        # The x and y coordinates are counted as for an np.array (Left Top point is the (0,0) mark)
        dataset["X_Bottom"].append(data_annotation["points"][1]["x"])
        dataset["Y_Bottom"].append(data_annotation["points"][1]["y"])
        dataset["X_Top"].append(data_annotation["points"][0]["x"])
        dataset["Y_Top"].append(data_annotation["points"][0]["y"])

        # Cropping an image plate
        image = images[f'Car_License_Image_{index}']
        car_image = Image.fromarray(image)
        # Top left point
        x_top = dataset['X_Top'][-1] * image.shape[1]
        y_top = dataset['Y_Top'][-1] * image.shape[0]
        # Bottom right point
        x_bottom = dataset['X_Bottom'][-1] * image.shape[1]
        y_bottom = dataset['Y_Bottom'][-1] * image.shape[0]
        plate_image = car_image.crop((x_top, y_top, x_bottom, y_bottom))
        plates[f'Car_License_Image_{index}'] = np.array(plate_image)

    mean = torch.mean(torch.tensor(means))
    std = torch.mean(torch.tensor(stds))
    image_statistics: Dict = {'mean': mean,
                              'std': std}

    # Saving the dictionaries and data frames
    np.save('images.npy', images)
    np.save('plates.npy', plates)
    np.save('image_statistics.npy', image_statistics)

    dataset: pd.DataFrame = pd.DataFrame(dataset)
    # Images height and width won't be helpful for our model training, and we've already saved their original values
    dataset.drop(["Image_height", "Image_width"], axis=1, inplace=True)
    dataset.to_csv('./Dataset/License_Plate_Dataset.csv', index=False)
