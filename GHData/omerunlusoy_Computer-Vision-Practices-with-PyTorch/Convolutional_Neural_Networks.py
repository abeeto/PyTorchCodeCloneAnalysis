import torch
import torch.nn as nn
import torch.nn.functional as Functional

from torchvision import datasets, transforms  # torchvision package contains many types of datasets (including MNIST dataset)

import numpy as np
import matplotlib.pyplot as plt

import requests                                                     # HTTP requests
import PIL.ImageOps
from PIL import Image                                               # Python Imaging Library

import os
import shutil

""" Convolutional Neural Networks """
# Convolutional Neural Networks are used for image recognition since they are more computational manageable
# more effective to recognize the useful patterns since it understands the spatial structure of inputs
# require lower quantity of parameters compared to artificial neural networks (regular nn)
# main similarities: input layer, fully connected layer (multilayer Perceptron)
# image recognition with ANN is not scalable with larger or colored images with multiple channels (like RGB)
# another problem of ANN is its tendency of overfitting

# ConvNN processes data that has a known grid like topology (image driven pattern recognition tasks)
# ConvNN has pooling layers which continuously reduces number of parameters and computations
# ConvNN comprised of 3 types of layers; convolutional layers, pooling layers, and fully connected layers (excluding input and output layers).
# after input layer, we have a series of convolutional and pooling layers followed by fully connected layers
# number of input and output parameters does not change
# uses softmax activation function in the output layer which outputs probabilities of belonging to a specific class

# inside the convolutional layer, all pixels are processed by convolutional filter (kernel, kernel matrix)
# kernel matrix slides over the image
# Stride: shift amount of kernel matrix (the bigger the stride, the smaller the corresponding feature map)
# Receptive Field: current position of kernel matrix
# multiply each image cell value with corresponding kernel matrix value and sum them all, then divide by the receptive field size
# result is put into corresponding feature map, shift the kernel matrix by stride and repeat until feature map is filled fully

# kernel matrix values are learned during the training through a gradient decent algorithm
# Translational Invariance: ConvNN has. Kernels detect types of features and if a kernel detects a feature in some part of given image, it is likely that it will
# be able to detect the same features in other parts of given image.
# each filter (kernel), detects a specific feature with its own feature map, more filters more feature detections (depth is number of feature maps)
# combining all feature maps gives the final output of the convolutional layer

# filtering RGB image, kernel has to be 3D (depth of the kernel equals to the depth of the image)

# Relu activation function will introduce non-linearity (more biological), (does not suffer from Vanishing Gradient Problem)

# Pooling layer shrink the image stack by reducing the dimensionality (reduces computational complexity), (avoids overfitting), (sum, average, max)
# pooling protects network from small translations of input
# lower layers corresponds to simple aspects of the image while higher levels corresponds to more sophisticated features
# Fully connected layers classifies the processed and flattened image (multilayer Perceptron)
# Training changes filter matrix values of convolutional layer and connection rates of fully connected layer
# Padding: extra gray border for filter to consider image edges (preserves image size)

# Many ConvNN architectures are available; LeNet, AlexNet, GoogleNet, ZFNet models, etc.
# This code will use LeNet model with 2 convolutional + pooling layers and 2 fully connected layers

# Dropout layer: randomly turns off some neurons to reduce overfitting, can be replaced between any two layers but better if it is placed at high number of parameters

########################################################################################################################

input_channel_num = 1                                                                               # gray scale
output_channel1_num = 20
output_channel2_num = 50
kernel_size = 5
stride_length = 1

image_size = 28
pool2_output_size = int((int((image_size - kernel_size + 1) / 2) - kernel_size + 1) / 2)
fc1_input_size = pool2_output_size * pool2_output_size * output_channel2_num
fc1_output_size = 500
class_num = 10

pooling_kernel_size = 2

dropout_rate = 0.5

batch_size = 100

learning_rate = 0.0001
epochs = 12

print_initial_dataset = True
print_internet_testset = True
print_testset = True
plot_loss_and_corrects = True
train_anyway = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")                             # specifies run device for more optimum runtime

save_path_name = 'saved_models_conv'

########################################################################################################################

# 1st dimension: color, 2nd dimension: width, 3rd dimension: height of image and pixels
def image_convert_to_numpy(tensor):
    image = tensor.clone().detach().cpu().numpy()                                                   # clones to tensor and transforms to numpy array. OR tensor.cpu().clone().detach().numpy()
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
            ax.set_title([labels[index].item()])
            # plt.savefig('trainset.jpg', dpi=500, bbox_inches='tight')
        else:
            ax.set_title("{} ({})".format(str(labels[index].item()), str(predictions[index].item())),
                         color=("green" if predictions[index] == labels[index] else "red"))
            # plt.savefig('testset.jpg', dpi=500, bbox_inches='tight')

    plt.show()


def get_internet_image(transform):
    url = 'https://images.homedepot-static.com/productImages/007164ea-d47e-4f66-8d8c-fd9f621984a2/svn/architectural-mailboxes-house-letters-numbers-3585b-5-64_1000.jpg'
    response = requests.get(url, stream=True)
    image = Image.open(response.raw)
    image = PIL.ImageOps.invert(image)                                          # convert colors (our NN is trained for black backgrounds)
    image = image.convert('1')                                                  # convert from RGB to binary black-white
    image = transform(image)                                                    # transform the image as we did to previous ones

    if print_internet_testset:
        plt.imshow(image_convert_to_numpy(image))
        plt.title('Image')
        # plt.savefig('image.jpg', dpi=500, bbox_inches='tight')
        plt.show()
    return image


def save_model(model):
    # hash every variable that matters to be sure that the saved model is the exact same
    hashed_vars = "conv_model_" + str(hash((input_channel_num, output_channel1_num, output_channel2_num, kernel_size, stride_length, image_size, fc1_output_size,
                            class_num, pooling_kernel_size, dropout_rate, batch_size, learning_rate, epochs, device)))

    # Path to save the model
    if not os.path.exists(save_path_name):
        os.mkdir(save_path_name)

    path = os.path.join(save_path_name, hashed_vars)
    if not os.path.isfile(path):
        torch.save(model, path)


def get_model():
    hashed_vars = "conv_model_" + str(
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
class LeNet(nn.Module):

    def __init__(self, init_weight_name=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channel_num, out_channels=output_channel1_num, kernel_size=kernel_size, stride=stride_length)
        self.conv2 = nn.Conv2d(in_channels=output_channel1_num, out_channels=output_channel2_num, kernel_size=kernel_size, stride=stride_length)
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
        x = Functional.relu(self.conv1(x))                                              # activation function is relu rather than sigmoid
        x = Functional.max_pool2d(x, pooling_kernel_size, pooling_kernel_size)
        x = Functional.relu(self.conv2(x))
        x = Functional.max_pool2d(x, pooling_kernel_size, pooling_kernel_size)
        x = x.view(-1, fc1_input_size)                                                  # x must be flattened before entering fully connected layer

        x = Functional.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)                                                                 # rather than the probability, we get score (raw output) for nn.CrossEntropyLoss
        return x

    def plot_loss_and_corrects_epoch(self, epochs, losses, corrects, validation_losses, validation_corrects):
        plt.title('Epoch vs Loss')
        plt.plot(range(epochs), losses, label="training loss")
        plt.plot(range(epochs), validation_losses, label="validation loss")
        plt.xlabel('epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(color='gray', linestyle='-', linewidth=0.2)
        # plt.savefig('Epoch vs Loss.jpg', dpi=500, bbox_inches='tight')
        plt.show()

        plt.title('Epoch vs Corrects')
        plt.plot(range(epochs), corrects, label="training accuracy")
        plt.plot(range(epochs), validation_corrects, label="validation accuracy")
        plt.xlabel('epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(color='gray', linestyle='-', linewidth=0.2)
        # plt.savefig('Epoch vs Corrects.jpg', dpi=500, bbox_inches='tight')
        plt.show()


########################################################################################################################

""" LeNet Sequential Class """
class LeNet_Sequential(nn.Module):

    def __init__(self):
        super().__init__()
        model = nn.Sequential()

        model.add_module('conv1', nn.Conv2d(in_channels=input_channel_num, out_channels=output_channel1_num, kernel_size=kernel_size, stride=stride_length))
        model.add_module('relu1', nn.ReLU())
        model.add_module('max1', nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=1))
        model.add_module('conv2', nn.Conv2d(in_channels=output_channel1_num, out_channels=output_channel2_num, kernel_size=kernel_size, stride=stride_length))
        model.add_module('relu2', nn.ReLU())
        model.add_module('max2', nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=1))
        model.add_module('lin1', nn.Linear(in_features=fc1_input_size, out_features=fc1_output_size))
        model.add_module('relu3', nn.ReLU())
        model.add_module('dropout1', nn.Dropout(dropout_rate))
        model.add_module('lin2', nn.Linear(in_features=fc1_output_size, out_features=class_num))

        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x

    def plot_loss_and_corrects_epoch(self, epochs, losses, corrects, validation_losses, validation_corrects):
        plt.title('Epoch vs Loss')
        plt.plot(range(epochs), losses, label="training loss")
        plt.plot(range(epochs), validation_losses, label="validation loss")
        plt.xlabel('epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(color='gray', linestyle='-', linewidth=0.2)
        # plt.savefig('Epoch vs Loss2.jpg', dpi=500, bbox_inches='tight')
        plt.show()

        plt.title('Epoch vs Corrects')
        plt.plot(range(epochs), corrects, label="training accuracy")
        plt.plot(range(epochs), validation_corrects, label="validation accuracy")
        plt.xlabel('epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(color='gray', linestyle='-', linewidth=0.2)
        # plt.savefig('Epoch vs Corrects2.jpg', dpi=500, bbox_inches='tight')
        plt.show()

########################################################################################################################

def train_network(model_conv, training_loader, criterion, optimizer):
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

        for images, labels in training_loader:                                      # for each epoch, iterate through each training batch (size of bach_size)

            images = images.to(device)
            labels = labels.to(device)

            # no need to flatten the images as we did in ANN since we are passing them to convolutional layers
            outputs = model_conv.forward(images)                                    # make a prediction (outputs of NN), (y_pred)
            # print(outputs.shape, labels.shape)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()                                                   # zeros (resets grad), (grads accumulates after each backward)
            loss.backward()                                                         # derivative (gradient) of loss
            optimizer.step()                                                        # optimizer recalculates params (can be called after loss.backward())

            _, predicted_classes = torch.max(outputs, 1)                            # gets the maximum output value for each output
            num_correct_predictions = torch.sum(predicted_classes == labels.data)   # predicted_classes == labels.data is something like [1 0 1 1 1 0]
            # print(outputs, predicted_classes, num_correct_predictions, labels)
            # num_correct_predictions = 4
            running_corrects += num_correct_predictions
            running_loss += loss.item()

        epoch_loss = running_loss / len(training_loader)
        losses.append(epoch_loss)  # average loss of each epoch is added to the losses

        epoch_accuracy = running_corrects / len(training_loader)
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

            validation_epoch_loss = validation_running_loss / len(validation_loader)
            validation_losses.append(validation_epoch_loss)

            validation_epoch_accuracy = validation_running_corrects / len(validation_loader)
            validation_corrects.append(validation_epoch_accuracy)

            print('epoch:', e + 1, 'validation loss: {:.4f}'.format(validation_epoch_loss), 'validation accuracy: {:.4f}'.format(validation_epoch_accuracy),
                  '\n')

    if plot_loss_and_corrects:
        model_conv.plot_loss_and_corrects_epoch(epochs, losses, corrects, validation_losses, validation_corrects)

# MAIN #################################################################################################################


# training dataset
transform = transforms.Compose([transforms.Resize((28, 28)),
                                transforms.ToTensor(),                                  # from (0, 255) intensity to (0, 1) probability
                                transforms.Normalize((0.5,), (0.5,))])                  # mean and center deviation to normalize (ranges from -1 to 1)

training_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
training_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)        # shuffle not to stuck in a local minimum

data_iter = iter(training_loader)
images, labels = data_iter.next()

images = images.to(device)
labels = labels.to(device)

if print_initial_dataset:
    show_images(images, labels)

# validation dataset (Test Model)
validation_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)   # no need to shuffle

# get model
model_conv = get_model()

# train model
if model_conv is None or train_anyway:
    model_conv = LeNet().to(device=device)

    # nn.CrossEntropyLoss loss function is used for multiclass classification (requires raw output)
    # nn.CrossEntropyLoss is combination of log_softmax() and NLLLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_conv.parameters(), lr=learning_rate)

    train_network(model_conv, training_loader, criterion, optimizer)

    # save model
    save_model(model_conv)

# plot predictions
data_iter = iter(validation_loader)
images, labels = data_iter.next()

images = images.to(device)
labels = labels.to(device)

outputs = model_conv.forward(images)
_, predicted_classes = torch.max(outputs, 1)

if print_testset:
    show_images(images, labels, predicted_classes)

# internet image
internet_image = get_internet_image(transform)

internet_image = internet_image.to(device)
internet_image = internet_image[0].unsqueeze(0).unsqueeze(0)
internet_image_output = model_conv.forward(internet_image)
_, internet_image_predicted_class = torch.max(internet_image_output, 1)
print("predicted class of internet image:", internet_image_predicted_class.item())


# epoch: 12 loss: 0.0305 accuracy: 99.0900
# epoch: 12 validation loss: 0.0362 validation accuracy: 98.8900

