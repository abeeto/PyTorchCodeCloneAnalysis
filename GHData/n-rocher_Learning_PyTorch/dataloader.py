import os
import cv2
import pyvips
from PIL import Image
import numpy as np

import torch
from torchvision.transforms import ToTensor, Normalize, Compose

AUDI_A2D2_CATEGORIES = {
    1: {"name": "Road", "color": [[180, 50, 180], [255, 0, 255]]},
    2: {"name": "Lane", "color": [[255, 193, 37], [200, 125, 210], [128, 0, 255]]},
    3: {"name": "Crosswalk", "color": [[210, 50, 115]]},
    4: {"name": "Curb", "color": [[128, 128, 0]]},
    5: {"name": "Sidewalk", "color": [[180, 150, 200]]},

    6: {"name": "Traffic Light", "color": [[0, 128, 255], [30, 28, 158], [60, 28, 100]]},
    7: {"name": "Traffic Sign", "color": [[0, 255, 255], [30, 220, 220], [60, 157, 199]]},

    8: {"name": "Person", "color": [[204, 153, 255], [189, 73, 155], [239, 89, 191]]},

    9: {"name": "Bicycle", "color": [[182, 89, 6], [150, 50, 4], [90, 30, 1], [90, 30, 30]]},
    10: {"name": "Bus", "color": []},
    11: {"name": "Car", "color": [[255, 0, 0], [200, 0, 0], [150, 0, 0], [128, 0, 0]]},
    12: {"name": "Motorcycle", "color": [[0, 255, 0], [0, 200, 0], [0, 150, 0]]},
    13: {"name": "Truck", "color": [[255, 128, 0], [200, 128, 0], [150, 128, 0], [255, 255, 0], [255, 255, 200]]},

    14: {"name": "Sky", "color": [[135, 206, 255]]},
    15: {"name": "Nature", "color": [[147, 253, 194]]},
    16: {"name": "Building", "color": [[241, 230, 255]]}
}

CATEGORIES_COLORS = {
    0: {"name": "Background", "color": [0, 0, 0]},
    1: {"name": "Road", "color": [75, 75, 75]},
    2: {"name": "Lane", "color": [255, 255, 255]},
    3: {"name": "Crosswalk", "color": [200, 128, 128]},
    4: {"name": "Curb", "color": [150, 150, 150]},
    5: {"name": "Sidewalk", "color": [244, 35, 232]},

    6: {"name": "Traffic Light", "color": [250, 170, 30]},
    7: {"name": "Traffic Sign", "color": [255, 255, 0]},

    8: {"name": "Person", "color": [255, 0, 0]},

    9: {"name": "Bicycle", "color": [88, 41, 0]},
    10: {"name": "Bus", "color": [255, 15, 147]},
    11: {"name": "Car", "color": [0, 255, 142]},
    12: {"name": "Motorcycle", "color": [0, 0, 230]},
    13: {"name": "Truck", "color": [75, 10, 170]},

    14: {"name": "Sky", "color": [135, 206, 255]},
    15: {"name": "Nature", "color": [107, 142, 35]},
    16: {"name": "Building", "color": [241, 230, 255]}
}


CLASS_LABELS = {
    0: "Background",
    1: "Road",
    2: "Lane",
    3: "Crosswalk",
    4: "Curb",
    5: "Sidewalk",
    6: "Traffic Light",
    7: "Traffic Sign",
    8: "Person",
    9: "Bicycle",
    10: "Bus",
    11: "Car",
    12: "Motorcycle",
    13: "Truck",
    14: "Sky",
    15: "Nature",
    16: "Building"
}

values = CATEGORIES_COLORS.values()
CLASS_COLORS = np.zeros((len(values), 3), dtype=np.uint8)
for i, data in enumerate(values):
    CLASS_COLORS[i] = data["color"]

class A2D2_Dataset(torch.utils.data.Dataset):

    def __init__(self, type, size=(418, 418)):

        assert type in ["testing", "training", "validation"], 'add the folder type of data "testing", "training", or "validation"'

        self.input_size = size

        self.dataset_folder = r"U:\\A2D2 Camera Semantic\\" + type + "\\"
        self.input_img_paths, self.target_img_paths = self.getData()

        self.CATEGORIES = AUDI_A2D2_CATEGORIES

        self.img_transform = Compose([ToTensor(), Normalize((122.15811035 / 255.0, 123.63384277 / 255.0, 125.46741699 / 255.0), (26.7605721 / 255.0, 35.98626225 / 255.0, 39.93803676 / 255.0))])

    @staticmethod
    def n_class():
        return len(CATEGORIES_COLORS)

    @staticmethod
    def static_data():
        return {
            "labels": CLASS_LABELS,
            "colors": CLASS_COLORS
        }

    def name(self):
        return "A2D2Dataset"

    def getData(self):
        '''
        Find files of A2D2
        '''
        data_image = []
        data_label = []

        camera_day_folders = [os.path.join(self.dataset_folder, item) for item in os.listdir(self.dataset_folder) if os.path.isdir(self.dataset_folder + item)]
        for folder in camera_day_folders:
            camera_files_folder = os.path.join(folder, "camera", "cam_front_center")
            label_files_folder = os.path.join(folder, "label", "cam_front_center")

            camera_files_files = [os.path.join(camera_files_folder, file) for file in os.listdir(camera_files_folder)]
            label_files_files = [os.path.join(label_files_folder, file) for file in os.listdir(label_files_folder)]

            data_image = data_image + camera_files_files
            data_label = data_label + label_files_files

        return data_image, data_label

    def __len__(self):
        return len(self.target_img_paths)

    def __getitem__(self, idx):

        input_img_path = self.input_img_paths[idx]
        target_img_path = self.target_img_paths[idx]

        # Loading Image

        # -> With PIL
        # img = Image.open(input_img_path).convert("RGB")
        # img = img.resize(self.input_size)

        # -> With OpenCV
        img = cv2.imread(input_img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_NEAREST)

        # -> With VIPS
        # img = pyvips.Image.new_from_file(input_img_path, access="sequential")
        # img = img.thumbnail_image(self.input_size[0], height=self.input_size[1], no_rotate= True)
        # img = img.numpy()
        # print(img.shape)
        # exit()
        # mem_img = img.write_to_memory()
        # img = np.frombuffer(mem_img, dtype=np.uint8).reshape(self.input_size[0], self.input_size[1], 3)

        # Loading Target

        # -> With PIL
        # target_raw = Image.open(target_img_path)
        # target_raw = target_raw.resize(self.input_size)
        # target_raw = np.asarray(target_raw)

        # -> With OpenCV
        target_raw = cv2.imread(target_img_path, cv2.IMREAD_COLOR)
        target_raw = cv2.cvtColor(target_raw, cv2.COLOR_BGR2RGB)
        target_raw = cv2.resize(target_raw, self.input_size, interpolation=cv2.INTER_NEAREST)

        # -> With VIPS
        # target_raw = pyvips.Image.new_from_file(target_img_path, access="sequential")
        # target_raw = target_raw.thumbnail_image(self.input_size[0], height=self.input_size[1])
        # target_raw = target_raw.numpy()
        # print(target_raw.shape)
        # mem_target_raw = target_raw.write_to_memory()
        # target_raw = np.frombuffer(mem_target_raw, dtype=np.uint8).reshape(self.input_size[0], self.input_size[1], 3)

        background = np.zeros(self.input_size[::-1])
        target = np.zeros((len(self.CATEGORIES) + 1,) + self.input_size[::-1])

        # For every categories in the list
        for id_category in range(1, len(self.CATEGORIES) + 1):

            data_category = self.CATEGORIES[id_category]

            for color in data_category["color"]:

                # We select pixels belonging to that category
                test = cv2.inRange(target_raw, tuple(color), tuple(color))

                # Add the value where it belongs to
                target[id_category] = target[id_category] + (test >= 1)

                # Write what's used for the background
                background = background + test

        target[0] = background == 0

        # Transform to pytorch tensor
        img = self.img_transform(img)
        target = torch.from_numpy(target)

        return img, target


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    training_dataset = A2D2_Dataset("training", size=(512, 400))

    for id in range(len(training_dataset)):
        img, target = training_dataset.__getitem__(id)

        img = img.numpy()
        target = target.numpy()

        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        imgplot = plt.imshow(img.transpose(1, 2, 0))
        ax.set_title('Image')

        ax = fig.add_subplot(1, 2, 2)
        imgplot = plt.imshow(np.argmax(target, axis=0))
        ax.set_title('Target')

        plt.show()
