import random
import os
import torch
import numpy as np
from dataloader import A2D2_Dataset
import cv2 

from models.ResUNet import ResUNet
from models.UNet import UNet
from dataloader import CLASS_COLORS
from torchvision.transforms import ToTensor, Normalize, Compose

import time

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

        folder = r"U:\A2D2 Camera Semantic\camera_lidar\20180810_150607\camera\cam_front_center\\"

        test_dataset = os.listdir(folder)

        img_transform = Compose([ToTensor(), Normalize((122.15811035 / 255.0, 123.63384277 / 255.0, 125.46741699 / 255.0), (26.7605721 / 255.0, 35.98626225 / 255.0, 39.93803676 / 255.0))])


        new_frame_time = 0
        prev_frame_time = 0


        for img_path in test_dataset:
            
            new_frame_time = time.time()
            
            img = cv2.imread(folder + img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

            tensor_img = img_transform(img_resized)
            tensor_img = torch.unsqueeze(tensor_img, dim=0).to(device)

            result = model(tensor_img)

            result = torch.squeeze(result, dim=0)
            result = torch.argmax(result, dim=0)
            result = result.cpu().numpy()

            print(result.shape)

            # Index -> Couleur 
            argmax_result_segmentation = np.expand_dims(result, axis=-1)
            print(argmax_result_segmentation.shape)
            segmentation = np.squeeze(np.take(CLASS_COLORS, argmax_result_segmentation, axis=0))
        
            # Fps calculation
            fps = 1 // (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            # Showing results
            cv2.imshow("ROAD_IMAGE", cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
            cv2.imshow("SEGMENTATION_IMAGE", cv2.cvtColor(segmentation, cv2.COLOR_RGB2BGR))

            print(fps)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break