import torchvision
import torch
from torch.utils.data import Dataset
import os
import torchvision.transforms as transforms

import numpy as np
from PIL import Image

full_class_dict = {
    "#ff0000": "Car 1",
    "#c80000": "Car 2",
    "#960000": "Car 3",
    "#800000": "Car 4",
    "#b65906": "Bicycle 1",
    "#963204": "Bicycle 2",
    "#5a1e01": "Bicycle 3",
    "#5a1e1e": "Bicycle 4",
    "#cc99ff": "Pedestrian 1",
    "#bd499b": "Pedestrian 2",
    "#ef59bf": "Pedestrian 3",
    "#ff8000": "Truck 1",
    "#c88000": "Truck 2",
    "#968000": "Truck 3",
    "#00ff00": "Small vehicles 1",
    "#00c800": "Small vehicles 2",
    "#009600": "Small vehicles 3",
    "#0080ff": "Traffic signal 1",
    "#1e1c9e": "Traffic signal 2",
    "#3c1c64": "Traffic signal 3",
    "#00ffff": "Traffic sign 1",
    "#1edcdc": "Traffic sign 2",
    "#3c9dc7": "Traffic sign 3",
    "#ffff00": "Utility vehicle 1",
    "#ffffc8": "Utility vehicle 2",
    "#e96400": "Sidebars",
    "#6e6e00": "Speed bumper",
    "#808000": "Curbstone",
    "#ffc125": "Solid line",
    "#400040": "Irrelevant signs",
    "#b97a57": "Road blocks",
    "#000064": "Tractor",
    "#8b636c": "Non-drivable street",
    "#d23273": "Zebra crossing",
    "#ff0080": "Obstacles / trash",
    "#fff68f": "Poles",
    "#960096": "RD restricted area",
    "#ccff99": "Animals",
    "#eea2ad": "Grid structure",
    "#212cb1": "Signal corpus",
    "#b432b4": "Drivable cobblestone",
    "#ff46b9": "Electronic traffic",
    "#eee9bf": "Slow drive area",
    "#93fdc2": "Nature object",
    "#9696c8": "Parking area",
    "#b496c8": "Sidewalk",
    "#48d1cc": "Ego car",
    "#c87dd2": "Painted driv. instr.",
    "#9f79ee": "Traffic guide obj.",
    "#8000ff": "Dashed line",
    "#ff00ff": "RD normal street",
    "#87ceff": "Sky",
    "#f1e6ff": "Buildings",
    "#60458f": "Blurred area",
    "#352e52": "Rain dirt"
}

train_sequences = [
    "20180807_145028",
    "20180925_112730",
    "20181008_095521",
    "20181016_125231",
    "20181107_133258",
    "20181108_091945",
    "20181108_141609",
    "20181204_170238",
    "20180810_142822",
    "20180925_124435",
    "20181016_082154",
    "20181107_132300",
    "20181107_133445",
    "20181108_103155",
    "20181204_135952",
    "20181204_191844",
    "20180925_101535",
    "20180925_135056",
    "20181016_095036",
    "20181107_132730",
    "20181108_084007"
]
test_sequences = [
    "20181108_123750",
    "20181204_154421"
]


def colour_to_class(colour, learning_map):  # converts the [r,g,b] array of a single pixel to the class index.
    hex = '#%02x%02x%02x' % tuple(colour)
    final = learning_map.get(full_class_dict.get(hex))
    if final == None:  # if the colour, for some reason, doesn't map correctly
        return 0   # this happens a lot, but I don't know why....
    return final


class AudiSegmentationDataset(Dataset):
    def __init__(self, path, learning_map, split="train", positional=True):
        self.path = path
        self.split = split
        self.labels = []
        self.images = []
        if split == "train":
            self.images = os.listdir(f"{path}/train/")
            self.labels = os.listdir(f"{path}/train_labels/")
        if split == "val":
            self.images = os.listdir(f"{path}/val/")
            self.labels = os.listdir(f"{path}/val_labels/")
            # sequences = test_sequences
        # for folder in sequences:
        #     l = f"{path}/{folder}/label/cam_front_center"
        #     for file in os.listdir(l):
        #         self.labels.append(os.path.join(l, file))
        #     c = f"{path}/{folder}/camera/cam_front_center"
        #     for file in os.listdir(c):
        #         if file.endswith(".png"):  # this directory has JSON and .png
        #             self.images.append(os.path.join(c, file))

        self.positional = positional
        self.learning_map = learning_map
        # print(self.images[0])
        assert len(self.images) == len(self.labels)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize distribution for all channels.
        ])
        print("Audi dataset with ", len(self.images), " initialized.")


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        assert 0 <= index <= len(self.images)
        width, height = 960, 640
        # print(self.images[index], index)
        # print(self.labels[index], index)
        image = Image.open(self.path+"/"+self.split+"/"+self.images[index]).resize((width, height))
        label = Image.open(self.path+"/"+self.split+"_labels/"+self.labels[index]).resize((width, height))

        image = image.crop((0, height // 2, width, height))  # crop to the lower half
        label = label.crop((0, height // 2, width, height))  # crop to the lower half

        # height, then width
        in_tensor = self.transforms(image)

        if self.positional:
            # these are positional encodings. The roadlines are more often found in certain parts of the image, so
            # the network needs to be given region information to learn that.
            # These channels are simply gradients from -1 to 1, one going horizontally and the other vertically.
            horizontal = torch.linspace(-1, 1, steps=width).unsqueeze(0)
            horizontal = horizontal.expand(height // 2, width).unsqueeze(0)

            vertical = torch.linspace(-1, 1, steps=height // 2).unsqueeze(1)
            vertical = vertical.expand(height // 2, width).unsqueeze(0)
            in_tensor = torch.cat((in_tensor, horizontal, vertical), dim=0)

        return in_tensor, np.apply_along_axis(colour_to_class, 2, np.array(label), self.learning_map)
