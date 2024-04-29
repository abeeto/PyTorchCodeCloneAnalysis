import random
import torch
import numpy as np
from dataloader import A2D2_Dataset

from models.ResUNet import ResUNet
from models.UNet import UNet
from dataloader import CLASS_COLORS

if __name__ == "__main__":

    # Constant
    IMG_SIZE = (512, 512)
    MODEL_PATH = "./ResUNet_size-512-512_features-16_epoch-7.pth"

    # Use GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Model initialisation
    model = ResUNet(3, len(CLASS_COLORS))
    model.to(device)

    # Loading the model
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # Evaluating the model
    model.eval()

    # Not doing training so no gradient calcul
    with torch.no_grad():

        test_dataset = A2D2_Dataset("validation", size=IMG_SIZE)

        import matplotlib.pyplot as plt

        index = list(range(len(test_dataset)))

        random.shuffle(index)

        for id in index:
            img, target = test_dataset.__getitem__(id)

            result = model(torch.unsqueeze(img, dim=0).to(device))
            result = torch.squeeze(result, dim=0)
            result = torch.argmax(result, dim=0)
            result = result.cpu().numpy()

            img = img.numpy()
            target = target.numpy()

            fig = plt.figure()
            ax = fig.add_subplot(1, 3, 1)
            imgplot = plt.imshow(img.transpose(1, 2, 0))
            ax.set_title('Image')

            ax = fig.add_subplot(1, 3, 2)
            imgplot = plt.imshow(result)
            ax.set_title('Prediction')

            ax = fig.add_subplot(1, 3, 3)
            imgplot = plt.imshow(np.argmax(target, axis=0))
            ax.set_title('Target')
            plt.show()






        # Reading videos
        # for video_filename in os.listdir(VIDEO_PATH):

        #     filename = os.path.join(VIDEO_PATH, video_filename)
        #     cap = cv2.VideoCapture(filename)

        #     new_frame_time = 0
        #     prev_frame_time = 0

        #     # Reading frames
        #     while(cap.isOpened()):

        #         ret, frame = cap.read()
        #         new_frame_time = time.time()

        #         if not ret:
        #             break
                
        #         # Formatting the frame
        #         img_resized = cv2.resize(frame, IMG_SIZE, interpolation=cv2.INTER_AREA)
               
        #         img_resized_mean = np.mean(img_resized, axis=(0, 1))
        #         img_resized_std= np.std(img_resized, axis=(0, 1))

        #         img_transform = Compose([ToTensor(), Normalize(img_resized_mean/255, img_resized_std/255)])
        #         tensor_img = img_transform(img_resized)
        #         tensor_img = torch.unsqueeze(tensor_img, dim=0).to(device)

        #         # Frame -> Infering with the model -> Argmax
        #         result = model(tensor_img)
        #         result = torch.squeeze(result, dim=0)
        #         result = torch.nn.functional.softmax(result, dim=0)
        #         result[result < 0.8] = 0
        #         result = torch.argmax(result, dim=0)
        #         result = result.cpu().numpy()

        #         # Index -> Couleur 
        #         argmax_result_segmentation = np.expand_dims(result, axis=-1)
        #         segmentation = np.squeeze(np.take(CLASS_COLORS, argmax_result_segmentation, axis=0))
            
        #         # Fps calculation
        #         fps = 1 // (new_frame_time - prev_frame_time)
        #         prev_frame_time = new_frame_time

        #         # Showing results
        #         cv2.imshow("ROAD_IMAGE", img_resized)
        #         cv2.imshow("SEGMENTATION_IMAGE", cv2.cvtColor(segmentation, cv2.COLOR_RGB2BGR))

        #         print(fps)

        #         if cv2.waitKey(1) & 0xFF == ord('q'):
        #             break

        #     cap.release()
