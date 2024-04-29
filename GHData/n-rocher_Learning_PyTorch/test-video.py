import os
import cv2
import time
import torch
import numpy as np
from torchvision.transforms import ToTensor, Compose, Normalize

from models.ResUNet import ResUNet
from models.UNet import UNet
from dataloader import CLASS_COLORS

if __name__ == "__main__":

    # Constant
    MODEL_PATH = "./ResUNet_size-400-400_epoch-5_v-argmax0.685.pth"
    VIDEO_PATH = r"F:\ROAD_VIDEO\Clip"

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

    # Image transformation
    
    # Not doing training so no gradient calcul
    with torch.no_grad():

        # Reading videos
        for video_filename in os.listdir(VIDEO_PATH):

            filename = os.path.join(VIDEO_PATH, video_filename)
            cap = cv2.VideoCapture(filename)

            new_frame_time = 0
            prev_frame_time = 0

            # Reading frames
            while(cap.isOpened()):

                ret, frame = cap.read()
                new_frame_time = time.time()

                if not ret:
                    break
                
                # Formatting the frame
                img_resized = cv2.resize(frame, IMG_SIZE, interpolation=cv2.INTER_AREA)
                img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
               
                img_resized_mean = np.mean(img_resized, axis=(0, 1))
                img_resized_std= np.std(img_resized, axis=(0, 1))

                img_transform = Compose([ToTensor(), Normalize(img_resized_mean/255, img_resized_std/255)])
                tensor_img = img_transform(img_resized)
                tensor_img = torch.unsqueeze(tensor_img, dim=0).to(device)

                # Frame -> Infering with the model -> Argmax
                result = model(tensor_img)
                result = torch.squeeze(result, dim=0)
                result = torch.nn.functional.softmax(result, dim=0)
                # result[result < 0.8] = 0
                result = torch.argmax(result, dim=0)
                result = result.cpu().numpy()

                # Index -> Couleur 
                argmax_result_segmentation = np.expand_dims(result, axis=-1)
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

            cap.release()
