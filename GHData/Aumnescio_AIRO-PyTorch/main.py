# ---------------------------
# ===       Imports       ===
# ---------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL

# PyTorch imports
import torch
from torch.serialization import validate_cuda_device
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

from torch.utils.tensorboard.writer import SummaryWriter

# --------------------------------
# ===       PyTorch Code       ===
# --------------------------------

# Script Start
print("PyTorch Script running...")

# What are tensors?
#   - Tensors are a specialized data structure that are very similar to arrays and matrices.
#   - Tensors can run on GPU's or other specialized hardware.

#   Random Tensor example:
#     tensor([[0.5926, 0.7179, 0.0880],
#             [0.7599, 0.4643, 0.6083]])

#   - It is basically a table (matrix) of values.
#   +--------+--------+--------+
#   | 0.5827 | 0.7179 | 0.0880 |
#   +--------+--------+--------+
#   | 0.7599 | 0.4643 | 0.6083 |
#   +--------+--------+--------+

#   - Tensors have a shape and a datatype.
#   - The shape of the tensor above is [3, 2]
#   - The datatype is torch.float32 (Essentially just a float value.)

#   - PyTorch supports many mathematical operations that can be run on Tensors.
#       - These operations can also utilize GPU's, so they can be very fast.


# What are neural networks?
#   - Neural networks are essentially a collection of functions which are executed on the input data.

# Training a neural network:

#   - Forward Propagation (Forward Pass)
#       - The input data is ran through each function.
#       - The final output value represents the guess of the neural network.
#           - For example, if presented with images of a cat and a dog,
#               and asked to label the output as one of these,
#               the output value will either be 'cat' or 'dog'.
#       - A single pass is not expected to reach the correct outcome.

#   - Backward Propagation (Backward Pass)
#       - Backward propagation is the process of traversing backwards from the output.
#       - I do not truly understand this, (Relatively complicated math stuff)
#           but the point is to collect information about the error of the guess,
#           and then optimize the parameters, so that the next guess will be more accurate.
#       - This process can be automated in PyTorch.

# Global variables
BATCH_SIZE: int = 4

# NOTE: This got me basically 100% very easily. I suppose it is most likely overfitting, but still was nice to see.
# EPOCHS: int = 72
# LEARNING_RATE: float = 0.01

EPOCHS: int = 20
LEARNING_RATE: float = 0.01

# Specify directory which holds dataset.
data_directory = r'./dataset-raw/'
shapes_directory = r'./dataset-raw/shapes/'
labels_path = r'./dataset-raw/annotations/labels.csv'
image_folder = r'./dataset-raw/image-folder/'
save_path = r'./model/neural_model.pth'

writer = SummaryWriter("runs/Reclipse/testing_tensorboard")

# Perform required transforms.
# - Set images to 32x32
# - Set images to be grayscale. (So that there is only one input channel)
# NOTE: Not sure if this should be tensored first or last...
data_transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.Grayscale(1),
        ToTensor(),
        transforms.Normalize((0.5), (0.5))      # I do not know if this is required for my dataset.
    ]
)

# Create PyTorch Dataset from the Data in the folder, using the predefined transforms.
# - NOTE: The folder structure has been changed so that this class should work.
transformed_dataset = datasets.ImageFolder(
    image_folder,
    transform = data_transform
)

# Use PyTorch DataLoader to load in the transformed data. (Also specify batch-size and shuffling.)
trainloader = DataLoader(
    transformed_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True
)

testloader = DataLoader(
    transformed_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True
)


# Defining Neural Network (Model)
class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        # NOTE: These values define a lot of the neural network, and they are manually set.
        # - The values are somewhat based on what the input values are.
        # - If our image is grayscale, the first convolutional layer input channels can be just 1.
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.fc1 = nn.Linear(12 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 60)
        self.out = nn.Linear(60, 10)

    # This method (function) defines a forward pass.
    # - Can't say I really understand this.
    # - It seems to utilize the previously defined layers. (As it should.)
    # NOTE: Apparently the backward function will be automatilly defined.
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

# Initializing network.
model = MyNeuralNetwork()

# Printing info about network layers.
print("Neural Network Layers: ")
print(model)

# Printing info about network paremeters.
network_parameters = list(model.parameters())
print(f"Len of params: {len(network_parameters)}")
print(f"Weights of layer 1: {network_parameters[0].size()}")

# Test print data and labels as tensors.
def print_data_as_tensors():
    for index, data in enumerate(trainloader, 0):
        print(index)
        input, labels = data
        print(f"Inputs: {input}")               # Inputs contains BATCH_SIZE images.
        print(f"Labels: {labels}")              # Labels also contains BATCH_SIZE labels. Hmm.
        print(f"InputShape: {input.size()}")
        print(f"LabelShape: {labels.size()}")

# Call the print. (Noisy output, disabled if not needed.)
# print_data_as_tensors()

classes = ('ellipse', 'rectangle', 'indexerror', 'indexerror')

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 6))
    for idx in np.arange(2):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

# Model Training Section -> Start

# Define loss function. 
criterion = nn.CrossEntropyLoss()                                    # Use CrossEntropyLoss as loss function.

# Define optimizer to use. (Pre-existing optimizer in PyTorch.)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)     # Stochastic gradient descent

# Training loop.
def train(number_of_epochs: int):
    log_step = 0
    for epoch in range(number_of_epochs):
        running_loss = 0.0

        # Loop over data.
        for i, data in enumerate(trainloader, 0):
            # Get the inputs; data is a list of [inputs, labels]
            input_image_batch, target_labels = data

            # Zero the gradients.
            optimizer.zero_grad()

            # Put the inputs through the model. (Neural Network)
            output = model(input_image_batch)           # Forward Pass

            # Calculate loss using loss function. Output and target as input. The labels are the targets.
            loss = criterion(output, target_labels)
            print(f"Loss: {loss}")
            loss.backward()                             # Backward Pass
            optimizer.step()                            # Gradient Descent

            # features = input_image_batch.reshape(input_image_batch.shape[0], -1)
            _, predictions = output.max(1)
            # class_labels = [classes_array[label] for label in predictions]
            num_correct = (predictions == target_labels).sum()
            running_training_accuracy = float(num_correct) / float(input_image_batch.shape[0])

            # Print statistics. (Optional)
            running_loss += loss.item()
            if i % 10 == 9:    # print every 50 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')

                # Log loss and accuracy to Tensorboard.
                writer.add_scalar("Training Loss",
                                running_loss / 10,
                                global_step=epoch * len(trainloader) + i)

                writer.add_scalar("Training Accuracy",
                                running_training_accuracy, 
                                global_step=epoch * len(trainloader) + i)

                # writer.add_embedding(features, metadata=class_labels,
                                    # label_img=input_image_batch, global_step=i)

                # Log images used in batches to Tensorboard.
                image_grid = torchvision.utils.make_grid(input_image_batch)
                writer.add_image("Images",
                                image_grid,
                                global_step=epoch * len(trainloader) + i)

                # ...log a Matplotlib Figure showing the model's predictions on a
                # random mini-batch
                writer.add_figure('predictions vs. actuals',
                                plot_classes_preds(model, input_image_batch, target_labels),
                                global_step=epoch * len(trainloader) + i)

                running_loss = 0.0


# Activate Training
train(EPOCHS)
print("Finished Training.")
torch.save(model.state_dict(), save_path)

model_to_validate = MyNeuralNetwork()
model_to_validate.load_state_dict(torch.load(save_path))

# Do some sort of validation testing.
def validate():

    validation_step = 0
    for i, data in enumerate(testloader, 0):

        image_batch, labels = data

        writer.add_figure('validation',
                        plot_classes_preds(model_to_validate, image_batch, labels),
                        global_step=validation_step)

        validation_step += 1


validate()



# Mostly irrelevant notes:

# Show amount of images in dataset.
# print("Image-count: ", len(images))

# Example of loading a single image into a variable:
# image = images[2][0]

# Example of visualizing a loaded-in image.
# plt.imshow(image, cmap="gray")

# Example of printing the size (dimensions) of the image.
# print("Image size: ", image.size())

# Print labels
# print(' '.join(f'{classes[labels[i]]:5s}' for i in range(BATCH_SIZE)))


