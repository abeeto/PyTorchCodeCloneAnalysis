import random
import os
from tkinter import E
import torch
import numpy as np
from dataloader import A2D2_Dataset
import cv2

from models.ResUNet import ResUNet
from models.UNet import UNet
from dataloader import CLASS_COLORS
from torchvision.transforms import ToTensor, Normalize, Compose

import time
from glob import glob

LIST_CAMERA = ["cam_front_center", "cam_front_left", "cam_front_right", "cam_side_left", "cam_side_right", "cam_rear_center"]


def open_image(path, img_size):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, img_size, interpolation=cv2.INTER_NEAREST)

    return img, img_resized

def open_cameras(images, transform, img_size):

    tensors = []
    data = []

    for camera_type in LIST_CAMERA:

        if camera_type in images:
            _, img_resized = open_image(images[camera_type], img_size)

            img_tensor = transform(img_resized)
            
            if camera_type == "cam_front_left":
                pass
                # img_resized = cv2.rectangle(img_resized, (200, 400), (512, 512), (0, 0, 0), -1)
            
                # img_resized[200:400, 200:400] = (0, 0, 0)
                # img_tensor[200:400, 200:400] = torch.Tensor([0, 0, 0])

            data.append(img_resized)
            tensors.append(img_tensor)
        else:
            tensors.append(torch.zeros((3,) + img_size))
            data.append(np.zeros(img_size + (3,)))

    return np.array(data, dtype=np.uint8), torch.stack(tensors)


if __name__ == "__main__":

    # Constant
    MODEL_PATH = "./ResUNet_size-400-400_epoch-10_v-argmax0.699.pth"

    # Use GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Loading the model
    checkpoint = torch.load(MODEL_PATH)
    IMG_SIZE = checkpoint["config"]["img_size"]
    features = checkpoint["config"]["features"]
    kernel_size = checkpoint["config"]["kernel_size"]

    # Model initialisation
    model = ResUNet(3, len(CLASS_COLORS), features=features, kernel_size=kernel_size)
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluating the model
    model.eval()

    # Not doing training so no gradient calcul
    with torch.no_grad():

        root_folder = r"U:\A2D2 Camera Semantic\camera_lidar\20180810_150607\camera\\"

        # Chargement des images :
        dict_images = {}

        for folder_cam in LIST_CAMERA:

            image_camera_folder = glob(root_folder + folder_cam + "\\" + "*.png")

            for image_path in image_camera_folder:

                id_image = image_path[-13:-4]

                if id_image not in dict_images:
                    dict_images[id_image] = {}

                dict_images[id_image][folder_cam] = image_path

        # Modele + inférence

        img_transform = Compose([ToTensor(), Normalize((122.15811035 / 255.0, 123.63384277 / 255.0, 125.46741699 / 255.0), (26.7605721 / 255.0, 35.98626225 / 255.0, 39.93803676 / 255.0))])

        new_frame_time = 0
        prev_frame_time = 0

        i = 0

        for images_folder in dict_images:
            
            i+=1

            if i % 30 == 0 :

                new_frame_time = time.time()

                multi_camera_data, multi_camera_tensors = open_cameras(dict_images[images_folder], img_transform, IMG_SIZE)

                result = model(multi_camera_tensors.to(device))

                result = torch.argmax(result, dim=1)
            
                result = result.cpu().numpy()

                # Fps calculation
                fps = 1 // (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time

                # Showing results

                for camera_index, camera_name in enumerate(LIST_CAMERA):
                    cv2.imshow(camera_name, cv2.cvtColor(multi_camera_data[camera_index], cv2.COLOR_RGB2BGR))

                    # Index -> Couleur
                    argmax_result_segmentation = np.expand_dims(result[camera_index], axis=-1)
                    segmentation = np.take(CLASS_COLORS, argmax_result_segmentation, axis=0)
                    segmentation = np.squeeze(segmentation)

                    cv2.imshow(f"{camera_name} - Segmentation", cv2.cvtColor(segmentation, cv2.COLOR_RGB2BGR))

                print(fps)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break