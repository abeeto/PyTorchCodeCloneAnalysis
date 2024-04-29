from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import timedelta, datetime
import cv2  # for video
from torchvision.utils import save_image


# this class includes some general supporting functions for CNN Transfer Learning
class Supporting_Functions:
    def __init__(self, log_filename, content_image_path, style_image_path, device, content_weight, style_weight, learning_rate, steps, storage_limit):
        super().__init__()
        self.log_filename = log_filename
        self.content_image_path = content_image_path
        self.style_image_path = style_image_path
        self.device = device
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.learning_rate = learning_rate
        self.steps = steps
        self.storage_limit = storage_limit

    def enter_log(self, text, header=False):
        file = open(self.log_filename, "a")

        if header:
            log_str = str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "\nContent Image: " + self.content_image_path + "\nStyle Image: " + self.style_image_path \
                      + "\ndevice: " + str(self.device) + ", content_weight: " + str(self.content_weight) + ", style_weight: " + str(self.style_weight) + ", learning_rate: " \
                      + str(self.learning_rate) + ", steps: " + str(self.steps) + ", storage_limit: " + str(self.storage_limit)
            file.write("\n\n" + log_str + "\n\n")

        file.write(text + "\n")
        file.close()

    # 1st dimension: color, 2nd dimension: width, 3rd dimension: height of image and pixels
    def image_convert_to_numpy(self, tensor):
        image = tensor.clone().detach().cpu().numpy()  # clones to tensor and transforms to numpy array. OR tensor.cpu().clone().detach().numpy()
        image = image.squeeze()
        image = image.transpose(1, 2, 0)
        # print(image.shape)                                                                            # (28, 28, 1)
        # denormalize image
        image = image * np.array((0.5,)) + np.array((0.5,))
        image = image.clip(0, 1)
        return image

    def load_image(self, path, max_size=600, shape=None):
        image = Image.open(path).convert('RGB')
        size = max(image.size)

        if size > max_size:
            size = max_size

        if shape is not None:
            size = shape

        # transform image to be compatible with the model
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

        image = transform(image).unsqueeze(0)  # to add extra dimensionality
        return image

    def create_video(self, images):
        frame_per_sec = 30
        frame_height, frame_width, _ = images[0].shape
        video = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), frame_per_sec, (frame_width, frame_height))

        for i in range(len(images)):
            cur_image = images[i]
            # make current image RGB
            cur_image = cur_image * 255
            cur_image = np.array(cur_image, dtype=np.uint8)
            cur_image = cv2.cvtColor(cur_image, cv2.COLOR_BGR2RGB)
            video.write(cur_image)

        video.release()
        # video is fully written and ready to be saved
        cv2.destroyAllWindows()
        self.enter_log('Video created.')

    def plot_images(self, images, nrows=1):
        names = ['Content Image', 'Style Image', 'Target Image']
        fig, axeslist = plt.subplots(ncols=int(len(images)/nrows), nrows=nrows)
        for i in range(len(images)):
            axeslist.ravel()[i].imshow(self.image_convert_to_numpy(images[i]))
            axeslist.ravel()[i].set_title(names[i])
            axeslist.ravel()[i].set_axis_off()
        plt.tight_layout()
        plt.show()

    def save_image(self, image, name):
        save_image(image, fp=name, normalize=True)
