import os
import torch
import numpy as np
import pandas as pd

from PIL import Image
from data_filtering import filtering


def _pre_processing():
    df: pd.DataFrame = pd.read_csv(r'./Dataset_web/license.csv')

    # Creating a Dataset data frame for Neural Network Model
    dataset = {"Image_name": [],
               "X_Bottom": df['bottom_x'],
               "Y_Bottom": df['bottom_y'],
               "X_Top": df['top_x'],
               "Y_Top": df['top_y']}

    # A dictionary that will map an image name (key) to a np.array which represents the Car image (value)
    images = {}

    # A dictionary that will map an image name (key) to a np.array which represents the Licence Plate image (value)
    plates = {}

    # Creating a directory to save all the dataset's images in
    if not os.path.isdir(r'./Dataset'):
        os.mkdir(r'./Dataset')
    dir_path = r'./Dataset/Cars License Plates'
    if not os.path.isdir(dir_path):
        os.mkdir(r'./Dataset/Cars License Plates')
    means, stds = [], []
    for root, dir_name, file_name in os.walk(r'./Dataset_web/Cars'):
        for file in file_name:
            if file.split('.')[-1] == 'jpg':
                image_name = file.split('.')[0]
                image_num = int(''.join(list(image_name)[3:]))

                image = Image.open(fr'./{root}/{file}')
                image = image.convert('RGB')
                image.save(fr'./Dataset/Cars License Plates/Car_License_Image_{image_num}.jpeg', "JPEG")
                images[f'Car_License_Image_{image_num}'] = np.array(image)

                image_tensor = torch.Tensor(np.array(image).flatten())
                means.append(torch.mean(image_tensor))
                stds.append(torch.std(image_tensor))

                image_num += 1

    mean = torch.mean(torch.tensor(means))
    std = torch.mean(torch.tensor(stds))
    image_statistics = {'mean': mean,
                        'std': std}

    for index in range(len(df)):
        dataset["Image_name"].append(f'Car_License_Image_{index}.jpeg')

        # Cropping an image plate
        image = images[f'Car_License_Image_{index}']
        car_image = Image.fromarray(image)

        # Top left point
        x_top = dataset['X_Top'].iloc[index] * image.shape[1]
        y_top = dataset['Y_Top'].iloc[index] * image.shape[0]

        # Bottom right point
        x_bottom = dataset['X_Bottom'].iloc[index] * image.shape[1]
        y_bottom = dataset['Y_Bottom'].iloc[index] * image.shape[0]
        plate_image = car_image.crop((x_top, y_top, x_bottom, y_bottom))
        plates[f'Car_License_Image_{index}'] = np.array(plate_image)

    # Saving the dictionaries and data frames
    np.save('./Dataset/images.npy', images)
    np.save('./Dataset/plates.npy', plates)
    np.save('./Dataset/image_statistics.npy', image_statistics)

    dataset = pd.DataFrame(dataset)

    # Filtering the Data
    # dataset = filtering(dataset)

    dataset.to_csv('./Dataset/License_Plate_Dataset.csv', index=False)
