import torch
import torch.nn as nn
import torch.nn.functional as Functional

from torchvision import datasets, transforms  # torchvision package contains many types of datasets (including MNIST dataset)

import numpy as np
import matplotlib.pyplot as plt

import requests  # HTTP requests
import PIL.ImageOps
from PIL import Image  # Python Imaging Library

import os
import shutil
import time
from datetime import timedelta
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd


""" CNN with Hyperparameter Tuning and Data Augmentation (CIFAR 10 Dataset) """

# Hyperparameter Tuning:
# - increase learning rate if the convergence of accuracy is slow.
# - add more convolutional layers to increase accuracy
# - decrease kernel size and add padding to reduce overfitting
#
# Data Augmentation:
# creating new data during the training process
# transforming or altering the existing dataset in useful ways to create new images (mimics larger dataset).
# new images are called Augmented Images.
# rotation, mirroring (flip), resizing, or combination
# Advance: de-texturizing, de-colorizing, edge enhanced, salient edge map
# creates variety, different perspective; allows it to extract relevant features more accurately.
# reduces OVERFITTING, increase GENERALIZATION.
#
# PyTorch allows us to implement Data Augmentation in Transforms

########################################################################################################################

input_channel_num = 3       # RGB
output_channel1_num = 16
output_channel2_num = 32
output_channel3_num = 64
kernel_size = 3
stride_length = 1
padding = 1

image_size = 32
pool3_output_size = 4
fc1_input_size = pool3_output_size * pool3_output_size * output_channel3_num
fc1_output_size = 500
class_num = 10

pooling_kernel_size = 2

dropout_rate = 0.5

batch_size = 100

learning_rate = 0.001
epochs = 15

print_initial_dataset = True
print_internet_testset = True
print_testset = True
plot_loss_and_corrects = True
train_anyway = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # specifies run device for more optimum runtime

save_path_name = 'saved_models_conv_ht'

classes = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]


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
        plt.imshow(image_convert_to_numpy(images[index]))

        if predictions is None:
            ax.set_title(classes[labels[index].item()])
            # plt.savefig('trainset.jpg', dpi=500, bbox_inches='tight')
        else:
            ax.set_title("{} ({})".format(str(classes[labels[index].item()]), str(classes[predictions[index].item()])),
                         color=("green" if predictions[index] == labels[index] else "red"))
            # plt.savefig('testset.jpg', dpi=500, bbox_inches='tight')

    plt.show()


def get_internet_image():
    url = 'https://www.aquariumofpacific.org/images/exhibits/Magnificent_Tree_Frog_900.jpg'
    response = requests.get(url, stream=True)
    image = Image.open(response.raw)
    # image = transform(image)  # transform the image as we did to previous ones

    if print_internet_testset:
        plt.imshow(image_convert_to_numpy(image))
        plt.title('Image')
        # plt.savefig('image.jpg', dpi=500, bbox_inches='tight')
        plt.show()
    return image


def save_model(model):
    # hash every variable that matters to be sure that the saved model is the exact same
    hashed_vars = "conv_ht_model_" + str(hash((input_channel_num, output_channel1_num, output_channel2_num, kernel_size, stride_length, image_size, fc1_output_size,
                                               class_num, pooling_kernel_size, dropout_rate, batch_size, learning_rate, epochs, device)))

    # Path to save the model
    if not os.path.exists(save_path_name):
        os.mkdir(save_path_name)

    path = os.path.join(save_path_name, hashed_vars)
    if not os.path.isfile(path):
        torch.save(model, path)


def get_model():
    hashed_vars = "conv_ht_model_" + str(
        hash((input_channel_num, output_channel1_num, output_channel2_num, kernel_size, stride_length, image_size, fc1_output_size,
              class_num, pooling_kernel_size, dropout_rate, batch_size, learning_rate, epochs, device)))

    if not os.path.exists(save_path_name):
        return None

    path = os.path.join(save_path_name, hashed_vars)
    if not os.path.isfile(path):
        return None

    model = torch.load(path)
    return model


########################################################################################################################

""" LeNet Class """
class CNN_HT_DA(nn.Module):

    def __init__(self, init_weight_name=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channel_num, out_channels=output_channel1_num, kernel_size=kernel_size, stride=stride_length, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=output_channel1_num, out_channels=output_channel2_num, kernel_size=kernel_size, stride=stride_length, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=output_channel2_num, out_channels=output_channel3_num, kernel_size=kernel_size, stride=stride_length, padding=padding)
        self.fc1 = nn.Linear(in_features=fc1_input_size, out_features=fc1_output_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(in_features=fc1_output_size, out_features=class_num)

        if init_weight_name is not None:
            self.initialize_initial_weights(init_weight_name)

    def initialize_initial_weights(self, init_weight_name):
        if init_weight_name.lower() == 'xavier_normal':
            nn_init = nn.init.xavier_normal_
        elif init_weight_name.lower() == 'xavier_uniform':
            nn_init = nn.init.xavier_uniform_
        elif init_weight_name.lower() == 'kaiming_normal':
            nn_init = nn.init.kaiming_normal_
        elif init_weight_name.lower() == 'kaiming_uniform':
            nn_init = nn.init.kaiming_uniform_
        else:
            raise ValueError(f'unknown initialization function: {init_weight_name}')

        for param in self.parameters():
            if len(param.shape) > 1:
                nn_init(param)

    def forward(self, x):
        x = Functional.relu(self.conv1(x))  # activation function is relu rather than sigmoid
        x = Functional.max_pool2d(x, pooling_kernel_size, pooling_kernel_size)
        x = Functional.relu(self.conv2(x))
        x = Functional.max_pool2d(x, pooling_kernel_size, pooling_kernel_size)
        x = Functional.relu(self.conv3(x))
        x = Functional.max_pool2d(x, pooling_kernel_size, pooling_kernel_size)
        x = x.view(-1, fc1_input_size)  # x must be flattened before entering fully connected layer

        x = Functional.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)  # rather than the probability, we get score (raw output) for nn.CrossEntropyLoss
        return x

    def plot_loss_and_corrects_epoch(self, epochs, losses, corrects, validation_losses, validation_corrects):
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

    def confusion_matrix(self, y_pred, y_true, classes, title='', save_fig=False):
        # Build confusion matrix

        cf_matrix = confusion_matrix(y_true, y_pred)
        cf_matrix = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 10, index=[i for i in classes], columns=[i for i in classes])
        plt.figure(figsize=(12, 7))
        plt.title(title)
        sn.heatmap(df_cm, annot=True, cmap="YlGnBu")
        if save_fig:
            plt.savefig('output.png')
        plt.show()


########################################################################################################################

def train_network(model_conv, training_loader, validation_loader, criterion, optimizer, classes):
    # iterations
    losses = []
    corrects = []
    validation_losses = []
    validation_corrects = []

    # each epoch
    for e in range(epochs):
        running_loss = 0.0
        running_corrects = 0.0
        validation_running_loss = 0.0
        validation_running_corrects = 0.0

        # each batch
        batch_num = 0
        y_pred = []
        y_true = []
        for images, labels in training_loader:  # for each epoch, iterate through each training batch (size of bach_size)

            batch_num += 1

            # print(batch_num, 'batch with', len(images))

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

            y_pred.extend(predicted_classes)
            y_true.extend(labels.data)

        epoch_loss = running_loss / len(training_loader)
        losses.append(epoch_loss)  # average loss of each epoch is added to the losses

        epoch_accuracy = running_corrects / len(training_loader)
        corrects.append(epoch_accuracy)

        print('epoch:', e + 1, 'loss: {:.4f}'.format(epoch_loss), 'accuracy: {:.4f}'.format(epoch_accuracy))

        model_conv.confusion_matrix(y_pred, y_true, classes, title='epoch' + str(e+1))




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

            validation_epoch_loss = validation_running_loss / len(validation_loader)
            validation_losses.append(validation_epoch_loss)

            validation_epoch_accuracy = validation_running_corrects / len(validation_loader)
            validation_corrects.append(validation_epoch_accuracy)

            print('epoch:', e + 1, 'validation loss: {:.4f}'.format(validation_epoch_loss), 'validation accuracy: {:.4f}'.format(validation_epoch_accuracy),
                  '\n')

    if plot_loss_and_corrects:
        model_conv.plot_loss_and_corrects_epoch(epochs, losses, corrects, validation_losses, validation_corrects)


# MAIN #################################################################################################################

# Data Augmentation (apply these transformations to training set only)
transform_train = transforms.Compose([transforms.Resize((32, 32)),                                              # resizes each image (pixels)
                                      transforms.RandomHorizontalFlip(),                                        # horizontal flip (lift to right)
                                      transforms.RandomRotation(10),                                            # rotation
                                      transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),                   # Affine Type Transformations (stretch, scale)
                                      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),     # changes color
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,), (0.5,))])

# Transformations for Validation Set
transform_validation = transforms.Compose([transforms.Resize((32, 32)),
                                transforms.ToTensor(),  # from (0, 255) intensity to (0, 1) probability
                                transforms.Normalize((0.5,), (0.5,))])  # mean and center deviation to normalize (ranges from -1 to 1)

# Training Dataset
training_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
training_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)  # shuffle not to stuck in a local minimum

# Validation Dataset
validation_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_validation)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)  # no need to shuffle

# check previous models
model_conv = get_model()
if model_conv is None or train_anyway:
    model_conv = CNN_HT_DA().to(device=device)

    # nn.CrossEntropyLoss loss function is used for multiclass classification (requires raw output)
    # nn.CrossEntropyLoss is combination of log_softmax() and NLLLoss()
    criterion = nn.CrossEntropyLoss()

    # Adam Optimizer
    optimizer = torch.optim.Adam(model_conv.parameters(), lr=learning_rate)

    # train CNN
    start_training_time = time.time()
    train_network(model_conv, training_loader, validation_loader, criterion, optimizer, training_dataset.classes)
    finish_training_time = time.time()
    print('Training Time: {:.4f}s', str(timedelta(seconds=finish_training_time - start_training_time)))

    # save model
    save_model(model_conv)


# get the first 20 images from validation dataset just to print
data_iter = iter(validation_loader)
images, labels = data_iter.next()

if print_initial_dataset:
    show_images(images, labels)

# predict images
outputs = model_conv.forward(images)
_, predicted_classes = torch.max(outputs, 1)

if print_testset:
    show_images(images, labels, predicted_classes)

# # internet image
# internet_image = get_internet_image(transform_validation)
# internet_image = internet_image.to(device)
# internet_image_output = model_conv.forward(internet_image)
# _, internet_image_predicted_class = torch.max(internet_image_output, 1)
# print("predicted class of internet image:", classes[internet_image_predicted_class.item()])
