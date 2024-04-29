import torch
import torch.nn as nn
from torchvision import datasets, transforms, models  # torchvision package contains many types of datasets (including MNIST dataset) and pre-trained models

import numpy as np
import matplotlib.pyplot as plt

import time
from datetime import timedelta, datetime
from Supporting_Functions import Supporting_Functions

""" CNN Style Transfer, VGG 19 """

# Style Transfer:
# - Image Content, Image Style
# - The aim of Style Transfer is to combine contents of a given image with the style of the completely different image.
#       - Input Image (Image Content) + Style Image (Image Style) -> Target Image
#
# - core content of the input image stays the same while its style is completely different.

# - Style Transfer can be done using mainly feature extraction from standard CNNs.
# 	- These features will be manipulated to extract style or content information (2 Gram Matrices will be matched).
#   - Feature Extraction: Extract outputs from model prematurely.
#
# - need 2 Datasets;
# 	- Input Image (Portraits/Selfies) Dataset
# 	- Style Image (Paintings) Dataset
#
# - Extra Resources:
# 	- Style Transfer - Styling Images with Convolutional Neural Networks (Medium),
# 		(https://gsurma.medium.com/style-transfer-styling-images-with-convolutional-neural-networks-7d215b58f461)
# 		(https://www.kaggle.com/greg115/style-transfer)
#
# 	- Style Transfer Deep Learning Algorithm (Kaggle), (https://www.kaggle.com/basu369victor/style-transfer-deep-learning-algorithm)
#
# 	- Artistic Neural Style Transfer using PyTorch (Kaggle),
# 		(https://www.kaggle.com/soumya044/artistic-neural-style-transfer-using-pytorch)
#
# 	- Neural Style Transfer using VGG19 (Kaggle), (https://www.kaggle.com/sayakdasgupta/neural-style-transfer-using-vgg19)

########################################################################################################################

# style content ratio: alpha represents content image weight and beta represents style image weight
content_weight = 1
style_weight = 1e6      # can play with it (1e1, 1e6)


# learning rate
learning_rate = 0.003

# step number
steps = 2100

# print variables
storage_limit = 300
print_per = 10
show_per = 50

# assign weight to each style layer for representation power (early layers have more style)
style_weights = {'conv1_1': 1.,
                'conv2_1': .75,
                'conv3_1': .2,
                'conv4_1': .2,
                'conv5_1': .2}

content_image_path = 'data/Images/HayleyWilliams.jpg'
style_image_path = 'data/Images/StarryNight.jpg'
log_filename = "log_CNN_Style_Transfer.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # specifies run device for more optimum runtime

########################################################################################################################

# returns pre-trained VGG19 model
def get_model():
    # VGG 19 pre-trained model
    model = models.vgg19(pretrained=True).features
    SF.enter_log('VGG 19 pre-trained model is created.')

    # freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # send model to GPU
    model.to(device=device)
    return model


# extracts the features from image using model
def get_features(image, model):
    # dictionary that holds the specific layer numbers where features will be extracted. You can play with them.
    # Conv1_1, Conv2_1, Conv3_1, Conv4_1, Conv4_2, Conv5_1
    layers = {'0': 'conv1_1',   # style extraction
              '5': 'conv2_1',   # style extraction
             '10': 'conv3_1',   # style extraction
             '19': 'conv4_1',   # style extraction
             '21': 'conv4_2',   # content extraction
             '28': 'conv5_1'}   # style extraction

    # dict that will store the extracted features
    features = {}
    # iterate through all layers and store the on es in the layers dict
    for name, layer in model._modules.items():
        # run image through all layers
        image = layer(image)
        if name in layers:
            features[layers[name]] = image
    return features


# Gram Matrix = V(T)*V  T: Transpose
def gram_matrix(tensor):
    # takes 4D image tensor
    # reshape the tensor
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h*w)
    gram = torch.mm(tensor, tensor.t())
    return gram


def train(model, content_image, style_image):
    # get content and style features
    content_features = get_features(content_image, model)
    style_features = get_features(style_image, model)

    # style features need one more step to be more useful (Gram Matrix)
    # applying Gram Matrix eliminates the remaining content information from style features
    # Gram Matrix = V(T)*V  T: Transpose
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    # define target image
    target_image = content_image.clone().requires_grad_(True).to(device=device)

    # training process images
    height, width, channels = SF.image_convert_to_numpy(target_image).shape
    images = np.empty(shape=(storage_limit, height, width, channels))

    # Adam Optimizer
    optimizer = torch.optim.Adam([target_image], lr=learning_rate)

    # training process
    SF.enter_log('Training begins...')
    iter = 0
    start_training_time = time.time()
    for ii in range(1, steps + 1):
        target_features = get_features(target_image, model)
        # calculate the content loss between content and target images using Mean Squared Error
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

        # calculate style loss iterating through 5 style layers
        style_loss = 0
        for style_layer in style_weights:
            # calculate target gram for cur layer
            target_feature = target_features[style_layer]
            target_gram = gram_matrix(target_feature)
            # get corresponding style gram from the precalculated list
            style_gram = style_grams[style_layer]
            current_style_loss = style_weights[style_layer] * torch.mean((target_gram - style_gram) ** 2)
            # normalize current_style_loss
            _, d, h, w = target_feature.shape
            style_loss += current_style_loss / (d * h * w)

        # optimizer will be used to optimize the parameters of the target image according to content and style losses
        # Style Aim: is to match the target gram matrix to the style gram matrix
        # Content Aim: is to match the target features (filtered image) to the content features
        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()  # reset optimizer
        total_loss.backward()
        optimizer.step()

        # data visualization throughout the training process
        # print period
        if ii % print_per == 0:
            finish_training_time = time.time()
            print_str = 'iteration: ' + str(ii) + ' loss: ' + str(total_loss.item()) + ' time passed: ' + str(timedelta(seconds=finish_training_time - start_training_time))
            print(print_str)
            SF.enter_log(print_str)

        # show image period
        if ii % show_per == 0:
            plt.imshow(SF.image_convert_to_numpy(target_image))
            plt.axis('off')
            plt.show()

        # store mid images period
        if ii % (steps / storage_limit) == 0:
            images[iter] = SF.image_convert_to_numpy(target_image)
            iter += 1

    SF.enter_log('Training completed.')
    return target_image, images


# MAIN #################################################################################################################
# initialize supporting functions
SF = Supporting_Functions(log_filename, content_image_path, style_image_path, device, content_weight, style_weight, learning_rate, steps, storage_limit)

# get content and style images
content_image = SF.load_image(content_image_path).to(device=device)
style_image = SF.load_image(style_image_path, shape=content_image.shape[-2:]).to(device=device)
SF.enter_log('Content and Style images are retrieved.', header=True)

# get VGG19 pre-trained model
model = get_model()

# plot content and style images
SF.plot_images([content_image, style_image])

# train target image
target_image, images = train(model, content_image, style_image)

# plot content, style, and target images
SF.plot_images([content_image, style_image, target_image])

# save target image
SF.save_image(target_image, 'target.jpg')

# create video
SF.create_video(images)

# END OF THE CODE

########################################################################################################################

# Sequential(
#   (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (1): ReLU(inplace=True)
#   (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (3): ReLU(inplace=True)
#   (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (6): ReLU(inplace=True)
#   (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (8): ReLU(inplace=True)
#   (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (11): ReLU(inplace=True)
#   (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (13): ReLU(inplace=True)
#   (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (15): ReLU(inplace=True)
#   (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (17): ReLU(inplace=True)
#   (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (20): ReLU(inplace=True)
#   (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (22): ReLU(inplace=True)
#   (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (24): ReLU(inplace=True)
#   (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (26): ReLU(inplace=True)
#   (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (29): ReLU(inplace=True)
#   (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (31): ReLU(inplace=True)
#   (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (33): ReLU(inplace=True)
#   (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (35): ReLU(inplace=True)
#   (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
# )
