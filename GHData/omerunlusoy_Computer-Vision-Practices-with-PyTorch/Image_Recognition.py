import PIL.ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as Functional
from torchvision import datasets, transforms                        # torchvision package contains many types of datasets (including MNIST dataset)
                                                                    # transformations applied to images to preprocess them before they are fed into a NN
import numpy as np
import matplotlib.pyplot as plt

import requests                                                     # HTTP requests
from PIL import Image                                               # Python Imaging Library

# MNIST Dataset (database of various handwritten digits), (10 classes)
# Softmax Activation Function (can deal with multiclass data sets)

# MNIST has this handwritten digit images in 28x28 pixel format
# this means 28x28 = 784 nodes for input layer, several hidden layers, and
# 10 nodes of output layer (one for each digit)
# the node that activates the highest activation value identifies the digit

# Generalization: ability to correctly classify new unlabeled data (test data)
# Underfitting: challenge to have small training error
# Overfitting: challenge to have small gap between training error and test error (DNN memorizes the training data so that it fails with test data)
# DNN needs to be deep and flexible enough
# to reduce overfitting: reduce depth, complexity, number of nodes, epochs; use larger datasets
# Regularization: techniques to reduce the generalization error.


# Variables ############################################################################################################

batch_size = 100

input_size = 28 * 28                                                             # corresponds to the pixel intensities (28*28 = 784)
H1 = 125                                                                         # nodes in first hidden layer, found experimentally
H2 = 65                                                                          # nodes in second hidden layer
output_size = 10                                                                 # output nodes, number of classes (digit classes)

learning_rate = 0.0005
epochs = 15

print_epoch_interval = 50

print_initial_dataset = True
print_internet_testset = True
print_testset = True
plot_loss_and_corrects = True

########################################################################################################################

# 1st dimension: color, 2nd dimension: width, 3rd dimension: height of image and pixels
def image_convert_to_numpy(tensor):
    image = tensor.clone().detach().numpy()                                     # clones to tensor and transforms to numpy array
    image = image.transpose(1, 2, 0)
    # print(image.shape)                                                        # (28, 28, 1)
    # denormalize image
    image = image * np.array((0.5, )) + np.array((0.5, ))
    image = image.clip(0, 1)
    return image


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


def show_images(images, labels, predictions=None):

    fig = plt.figure(figsize=(25, 4))

    for index in np.arange(20):
        ax = fig.add_subplot(2, 10, index + 1, xticks=[], yticks=[])
        plt.imshow(image_convert_to_numpy(images[index]))

        if predictions is None:
            ax.set_title([labels[index].item()])
            # plt.savefig('initialset_images.jpg', dpi=500, bbox_inches='tight')
        else:
            ax.set_title("{} ({})".format(str(labels[index].item()), str(predictions[index].item())), color=("green" if predictions[index] == labels[index] else "red"))
            # plt.savefig('testset_images.jpg', dpi=500, bbox_inches='tight')

    plt.show()


########################################################################################################################

""" Image Recognition (Classifier) Class """
class Image_Classifier(nn.Module):

    def __init__(self, input_size, H1, H2, output_size):
        super().__init__()
        self.linear1 = nn.Linear(in_features=input_size, out_features=H1)
        self.linear2 = nn.Linear(in_features=H1, out_features=H2)
        self.linear3 = nn.Linear(in_features=H2, out_features=output_size)


    def forward(self, x):
        x = Functional.relu(self.linear1(x))                                    # activation function is relu rather than sigmoid
        x = Functional.relu(self.linear2(x))
        x = self.linear3(x)                                                     # rather than the probability, we get score (raw output) for nn.CrossEntropyLoss
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
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(color='gray', linestyle='-', linewidth=0.2)
        # plt.savefig('Epoch vs Corrects.jpg', dpi=500, bbox_inches='tight')
        plt.show()

########################################################################################################################


# training dataset
transform = transforms.Compose([transforms.Resize((28, 28)),
                                transforms.ToTensor(),                          # from (0, 255) intensity to (0, 1) probability
                                transforms.Normalize((0.5, ), (0.5, ))])        # mean and center deviation to normalize (ranges from -1 to 1)

training_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
training_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)        # shuffle not to stuck in a local minimum

data_iter = iter(training_loader)
images, labels = data_iter.next()
if print_initial_dataset:
    show_images(images, labels)


# validation dataset (Test Model)
validation_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)        # no need to shuffle


# model
model_classifier = Image_Classifier(input_size, H1, H2, output_size)
# print(model_classifier)                                                       # Image_Classifier(
                                                                                #   (linear1): Linear(in_features=784, out_features=125, bias=True)
                                                                                #   (linear2): Linear(in_features=125, out_features=65, bias=True)
                                                                                #   (linear3): Linear(in_features=65, out_features=10, bias=True))


# nn.CrossEntropyLoss loss function is used for multiclass classification (requires raw output)
# nn.CrossEntropyLoss is combination of log_softmax() and NLLLoss()
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model_classifier.parameters(), lr=learning_rate)

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

    for images, labels in training_loader:                                          # for each epoch, iterate through each training batch (size of bach_size)

        images = images.view(images.shape[0], -1)                                   # flatten the inputs (images)

        outputs = model_classifier.forward(images)                                  # make a prediction (outputs of NN), (y_pred)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()                                                       # zeros (resets grad), (grads accumulates after each backward)
        loss.backward()                                                             # derivative (gradient) of loss
        optimizer.step()                                                            # optimizer recalculates params (can be called after loss.backward())

        _, predicted_classes = torch.max(outputs, 1)                                # gets the maximum output value for each output
        num_correct_predictions = torch.sum(predicted_classes == labels.data)       # predicted_classes == labels.data is something like [1 0 1 1 1 0]
                                                                                    # num_correct_predictions = 4
        running_corrects += num_correct_predictions
        running_loss += loss.item()

    epoch_loss = running_loss / len(training_loader)
    losses.append(epoch_loss)                                                       # average loss of each epoch is added to the losses

    epoch_accuracy = running_corrects / len(training_loader)
    corrects.append(epoch_accuracy)

    print('epoch:', e+1, 'loss: {:.4f}'.format(epoch_loss), 'accuracy: {:.4f}'.format(epoch_accuracy))

    with torch.no_grad():
        for validation_images, validation_labels in validation_loader:

            validation_images = validation_images.view(images.shape[0], -1)
            validation_outputs = model_classifier.forward(validation_images)
            validation_loss = criterion(validation_outputs, validation_labels)

            _, validation_predicted_classes = torch.max(validation_outputs, 1)
            validation_num_correct_predictions = torch.sum(validation_predicted_classes == validation_labels.data)
            validation_running_corrects += validation_num_correct_predictions
            validation_running_loss += validation_loss.item()

        validation_epoch_loss = validation_running_loss / len(validation_loader)
        validation_losses.append(validation_epoch_loss)

        validation_epoch_accuracy = validation_running_corrects / len(validation_loader)
        validation_corrects.append(validation_epoch_accuracy)

        print('epoch:', e+1, 'validation loss: {:.4f}'.format(validation_epoch_loss), 'validation accuracy: {:.4f}'.format(validation_epoch_accuracy), '\n')

if plot_loss_and_corrects:
    model_classifier.plot_loss_and_corrects_epoch(epochs, losses, corrects, validation_losses, validation_corrects)               # best: epoch: 11 loss: 0.0458 accuracy: 98.5050

# at the point validation error becomes higher than the training error, overfitting begins.


# internet image
internet_image = get_internet_image(transform)

internet_image = internet_image.view(internet_image.shape[0], -1)
internet_image_output = model_classifier.forward(internet_image)
_, internet_image_predicted_class = torch.max(internet_image_output, 1)
print("predicted class of internet image:", internet_image_predicted_class.item())


# plot predictions
data_iter = iter(validation_loader)
images, labels = data_iter.next()
images_flatten = images.view(images.shape[0], -1)

outputs = model_classifier.forward(images_flatten)
_, predicted_classes = torch.max(outputs, 1)

if print_testset:
    show_images(images, labels, predicted_classes)

