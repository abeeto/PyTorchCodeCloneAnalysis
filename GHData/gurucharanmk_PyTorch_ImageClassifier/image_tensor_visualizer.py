import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision


class ImageTensorVisualizer(object):
    def __init__(self, class_names, mean, std_dev, class_names_dict=None):
        self.class_names = class_names
        self.class_names_dict = class_names_dict
        self.mean = np.array(mean)
        self.std_dev = np.array(std_dev)

    def denormalize(self, input_image_tensor):
        tensor_to_display = input_image_tensor.numpy().transpose((1, 2, 0))
        tensor_to_display = self.std_dev * tensor_to_display + self.mean
        tensor_to_display = np.clip(tensor_to_display, 0, 1)
        return tensor_to_display

    def _imshow(self, input_tensor):
        # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
        tensor_to_display = self.denormalize(input_tensor)
        plt.imshow(tensor_to_display)
        plt.title("Summary of the batch")
        plt.pause(0.001)  # pause a bit so that plots are updated

    def batch_overview(self, image_tensor):
        image_tensor_grid = torchvision.utils.make_grid(image_tensor)
        self._imshow(image_tensor_grid)

    def imshow_subplot(
            self,
            image_tensor,
            title,
            other_info=None,
            row=4,
            col=8):
        fig = plt.figure(figsize=(22, 12))
        ax = []
        total_images = title.shape[0]
        for i in range(total_images):
            individual_image_tensor = image_tensor[i]
            tensor_to_display = self.denormalize(individual_image_tensor)
            ax.append(fig.add_subplot(row, col, i + 1))
            if self.class_names_dict:
                class_name_in_string = self.class_names_dict[self.class_names[title[i]]]
            else:
                class_name_in_string = self.class_names[title[i]]
            if other_info:
                class_name_in_string += other_info[i]
            ax[-1].set_title(class_name_in_string)  # set title
            plt.imshow(tensor_to_display)
        plt.show()
