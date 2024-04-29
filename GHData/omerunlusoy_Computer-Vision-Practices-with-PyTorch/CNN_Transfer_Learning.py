import torch
import torch.nn as nn
import torch.nn.functional as Functional

from torchvision import datasets, transforms, models  # torchvision package contains many types of datasets (including MNIST dataset) and pre-trained models

import numpy as np
import matplotlib.pyplot as plt

import requests  # HTTP requests
import PIL.ImageOps
from PIL import Image  # Python Imaging Library

import os
import shutil
import time
from datetime import timedelta


""" CNN with Transfer Learning, AlexNet (Ants and Bees Dataset) """

# Transfer Learning:
# - Usage of Pre-trained Sophisticated Models
# - Reasons:
#   - To increase accuracy and decrease training time.
#   - If NOT ENOUGH Labeled Data
#   - If similar model already exists
# - FREEZE parameters of some layers during training

# Convolutional Neural Networks consist of 2 parts;
# Feature Extraction Part = Convolutional Layers + Max Pooling Layers
# Classifier Part = Fully Connected Layers + Output Layer
# Freezing feature extraction part all together assumes that you have the same classes with the pre-training set.
# If you have different classes, freeze some portion of the feature extraction part.
# Freeze the first portion since feature extraction goes from general to specific.

########################################################################################################################

epochs = 5
learning_rate = 0.0001
batch_size = 20

print_initial_dataset = True
print_internet_testset = True
print_testset = True
plot_loss_and_corrects = True
train_anyway = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # specifies run device for more optimum runtime

save_path_name = 'saved_models_conv_tl'

classes = ["Ant", "Bee"]    # order is important


########################################################################################################################

# 1st dimension: color, 2nd dimension: width, 3rd dimension: height of image and pixels
def image_convert_to_numpy(tensor):
    image = tensor.clone().detach().cpu().numpy()  # clones to tensor and transforms to numpy array. OR tensor.cpu().clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    # print(image.shape)                                                                            # (28, 28, 1)
    # denormalize image
    image = image * np.array((0.5,)) + np.array((0.5,))
    image = image.clip(0, 1)
    return image


def show_images(images, labels, predictions=None):
    fig = plt.figure(figsize=(25, 4))

    for index in np.arange(20):
        ax = fig.add_subplot(2, 10, index + 1, xticks=[], yticks=[])
        # print(image_convert_to_numpy(images[index]).shape)
        plt.imshow(image_convert_to_numpy(images[index]))

        if predictions is None:
            ax.set_title(classes[labels[index].item()])
            # plt.savefig('trainset.jpg', dpi=500, bbox_inches='tight')
        else:
            ax.set_title("{} ({})".format(str(classes[labels[index].item()]), str(classes[predictions[index].item()])),
                         color=("green" if predictions[index] == labels[index] else "red"))
            # plt.savefig('testset.jpg', dpi=500, bbox_inches='tight')

    plt.show()


def save_model(model):
    # hash every variable that matters to be sure that the saved model is the exact same
    hashed_vars = "conv_tl_model_" + str(hash((batch_size, learning_rate, epochs, device)))

    # Path to save the model
    if not os.path.exists(save_path_name):
        os.mkdir(save_path_name)

    path = os.path.join(save_path_name, hashed_vars)
    if not os.path.isfile(path):
        torch.save(model, path)


def get_model():
    hashed_vars = "conv_tl_model_" + str(hash((batch_size, learning_rate, epochs, device)))

    if not os.path.exists(save_path_name):
        return None

    path = os.path.join(save_path_name, hashed_vars)
    if not os.path.isfile(path):
        return None

    model = torch.load(path)
    return model


def plot_loss_and_corrects_epoch(epochs, losses, corrects, validation_losses, validation_corrects):
    plt.title('Epoch vs Loss')
    plt.plot(range(epochs), losses, label="training loss")
    plt.plot(range(epochs), validation_losses, label="validation loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(color='gray', linestyle='-', linewidth=0.2)
    # plt.savefig('Epoch vs Loss.jpg', dpi=500, bbox_inches='tight')
    plt.show()

    plt.title('Epoch vs Corrects')
    plt.plot(range(epochs), corrects, label="training accuracy")
    plt.plot(range(epochs), validation_corrects, label="validation accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Correct')
    plt.legend()
    plt.grid(color='gray', linestyle='-', linewidth=0.2)
    # plt.savefig('Epoch vs Corrects.jpg', dpi=500, bbox_inches='tight')
    plt.show()


########################################################################################################################

def train_network(model_conv, training_loader, validation_loader, criterion, optimizer):
    # iterations
    losses = []
    corrects = []
    validation_losses = []
    validation_corrects = []

    for e in range(epochs):
        running_loss = 0.0
        running_corrects = 0.0
        validation_running_loss = 0.0
        validation_running_corrects = 0.0

        for images, labels in training_loader:  # for each epoch, iterate through each training batch (size of bach_size)

            images = images.to(device)
            labels = labels.to(device)

            # no need to flatten the images as we did in ANN since we are passing them to convolutional layers
            outputs = model_conv.forward(images)  # make a prediction (outputs of NN), (y_pred)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()  # zeros (resets grad), (grads accumulates after each backward)
            loss.backward()  # derivative (gradient) of loss
            optimizer.step()  # optimizer recalculates params (can be called after loss.backward())

            _, predicted_classes = torch.max(outputs, 1)  # gets the maximum output value for each output
            num_correct_predictions = torch.sum(predicted_classes == labels.data)  # predicted_classes == labels.data is something like [1 0 1 1 1 0]
            # num_correct_predictions = 4
            running_corrects += num_correct_predictions
            running_loss += loss.item()

        epoch_loss = running_loss / len(training_loader.dataset)
        losses.append(epoch_loss)  # average loss of each epoch is added to the losses

        epoch_accuracy = running_corrects / len(training_loader.dataset)
        corrects.append(epoch_accuracy)

        print('epoch:', e + 1, 'loss: {:.4f}'.format(epoch_loss), 'accuracy: {:.4f}'.format(epoch_accuracy))

        with torch.no_grad():
            for validation_images, validation_labels in validation_loader:
                validation_images = validation_images.to(device)
                validation_labels = validation_labels.to(device)

                validation_outputs = model_conv.forward(validation_images)
                validation_loss = criterion(validation_outputs, validation_labels)

                _, validation_predicted_classes = torch.max(validation_outputs, 1)
                validation_num_correct_predictions = torch.sum(validation_predicted_classes == validation_labels.data)
                validation_running_corrects += validation_num_correct_predictions
                validation_running_loss += validation_loss.item()

            validation_epoch_loss = validation_running_loss / len(validation_loader.dataset)
            validation_losses.append(validation_epoch_loss)

            validation_epoch_accuracy = validation_running_corrects / len(validation_loader.dataset)
            validation_corrects.append(validation_epoch_accuracy)

            print('epoch:', e + 1, 'validation loss: {:.4f}'.format(validation_epoch_loss), 'validation accuracy: {:.4f}'.format(validation_epoch_accuracy),
                  '\n')

    if plot_loss_and_corrects:
        plot_loss_and_corrects_epoch(epochs, losses, corrects, validation_losses, validation_corrects)


# MAIN #################################################################################################################

# Data Augmentation (apply these transformations to training set only)
transform_train = transforms.Compose([transforms.Resize((224, 224)),                                              # resizes each image (pixels)
                                      transforms.RandomHorizontalFlip(),                                          # horizontal flip (lift to right)
                                      # random rotation hinders the performance
                                      transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),                     # Affine Type Transformations (stretch, scale)
                                      transforms.ColorJitter(brightness=1, contrast=1, saturation=1),             # changes color (this time, use 1)
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,), (0.5,))])

# Transformations for Validation Set
transform_validation = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),  # from (0, 255) intensity to (0, 1) probability
                                transforms.Normalize((0.5,), (0.5,))])  # mean and center deviation to normalize (ranges from -1 to 1)

# Training Dataset
training_dataset = datasets.ImageFolder('data/ants_and_bees/train', transform=transform_train)
training_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)  # shuffle not to stuck in a local minimum

# Validation Dataset
validation_dataset = datasets.ImageFolder('data/ants_and_bees/val', transform=transform_validation)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)  # no need to shuffle

# check previous models
model_conv = get_model()
if model_conv is None or train_anyway:

    # AlexNet pre-trained model
    # model_conv = models.alexnet(pretrained=True)
    model_conv = models.vgg16(pretrained=True)

    # freeze parameters
    for param in model_conv.features.parameters():
        param.requires_grad = False

    # update the output layer
    n_inputs = model_conv.classifier[6].in_features         # 4096
    new_output_layer = nn.Linear(n_inputs, len(classes))
    model_conv.classifier[6] = new_output_layer

    # send model to GPU
    model_conv.to(device=device)

    # nn.CrossEntropyLoss loss function is used for multiclass classification (requires raw output)
    # nn.CrossEntropyLoss is combination of log_softmax() and NLLLoss()
    criterion = nn.CrossEntropyLoss()

    # Adam Optimizer
    optimizer = torch.optim.Adam(model_conv.parameters(), lr=learning_rate)

    # train CNN
    start_training_time = time.time()
    train_network(model_conv, training_loader, validation_loader, criterion, optimizer)
    finish_training_time = time.time()
    print('Training Time:', str(timedelta(seconds=finish_training_time - start_training_time)))

    # save model
    save_model(model_conv)


# get the first 20 images from validation dataset just to print
data_iter = iter(training_loader)
images, labels = data_iter.next()

if print_initial_dataset:
    show_images(images, labels)

# predict images
outputs = model_conv.forward(images)
_, predicted_classes = torch.max(outputs, 1)

if print_testset:
    show_images(images, labels, predicted_classes)

# AlexNet(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
#     (1): ReLU(inplace=True)
#     (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#     (4): ReLU(inplace=True)
#     (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (7): ReLU(inplace=True)
#     (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (9): ReLU(inplace=True)
#     (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (11): ReLU(inplace=True)
#     (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
#   (classifier): Sequential(
#     (0): Dropout(p=0.5, inplace=False)
#     (1): Linear(in_features=9216, out_features=4096, bias=True)
#     (2): ReLU(inplace=True)
#     (3): Dropout(p=0.5, inplace=False)
#     (4): Linear(in_features=4096, out_features=4096, bias=True)
#     (5): ReLU(inplace=True)
#     (6): Linear(in_features=4096, out_features=1000, bias=True)       # new out_features=2
#   )
# )
