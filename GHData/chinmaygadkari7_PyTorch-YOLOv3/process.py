import cv2
import numpy as np
import torch
from PIL import Image

def letterbox_image(image, dimention=416):
    '''resize image with unchanged aspect ratio using padding'''
    height, width = image.shape[0], image.shape[1]
    new_width = int(width * min(dimention/width, dimention/height))
    new_height = int(height * min(dimention/width, dimention/height))
    resized_image = cv2.resize(image, (new_width,new_height), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((dimention, dimention, 3), 128)
    start_x = (dimention - new_width) // 2
    start_y = (dimention - new_height) // 2
    end_x = start_x + new_width
    end_y = start_y + new_height
    canvas[start_y:end_y,start_x:end_x, :] = resized_image

    return canvas

def preprocess_image(image, dimentions=416):
    image = cv2.imread(image)
    #height, width = image.shape[0], image.shape[1]
    letterboxed_image = letterbox_image(image, dimentions)
    processed_image = letterboxed_image.transpose((2,0,1))
    processed_image = torch.from_numpy(processed_image).float().div(255.0).unsqueeze(0)
    return processed_image, image
