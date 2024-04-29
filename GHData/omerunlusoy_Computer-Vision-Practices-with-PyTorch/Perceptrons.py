import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets                                                # to create linearly separable data set

""" Perceptron """
# Single Layered Neural Network (most basic NN)
# Back Propagation
# Simple Neural Network includes 1 input, 1 hidden, and 1 output layers
# Deep (Learning) Neural Network includes more than 1 hidden layer

# first, create linearly separable data set using sklearn
# then, create Perceptron based Neural Networks
# then, train it to learn how to fit our data set (separate our data into two distinct classes)
# do this by using optimization algorithm known as gradient descent


########################################################################################################################

""" Perceptron Class """
class Perceptron(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=output_size)


    def forward(self, x):
        prediction = torch.sigmoid(self.linear(x))
        return prediction


    def predict(self, x):
        pred = self.forward(x).item()
        if pred >= 0.5:
            return 1
        else:
            return 0


    def get_params(self):
        [w, b] = self.parameters()                                             # returns the weights and bias of given model
        weight1 = w[0][0].item()
        weight2 = w[0][1].item()
        bias = b[0].item()
        return weight1, weight2, bias


    def plot_fit_and_dataset(self, title, coordinates, labels):
        plt.title(title)
        w1, w2, b = self.get_params()

        # 0 = w1*x1 + w2*x2 + b
        x1 = np.array([-2.0, 2.0])
        x2 = - (w1*x1 + b) / w2

        # plot fit
        plt.plot(x1, x2, 'r')

        # plot dataset
        plt.scatter(coordinates[labels == 0, 0], coordinates[labels == 0, 1])  # first center samples
        plt.scatter(coordinates[labels == 1, 0], coordinates[labels == 1, 1])  # second center samples

        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.grid(color='gray', linestyle='-', linewidth=0.2)
        # plt.savefig(title + '.jpg', dpi=500, bbox_inches='tight')
        plt.show()


    def plot_loss_epoch(self, epochs, losses):
        plt.title('Epoch vs Loss')
        plt.plot(range(epochs), losses)
        plt.xlabel('epoch')
        plt.ylabel('Loss')
        plt.grid(color='gray', linestyle='-', linewidth=0.2)
        # plt.savefig('Epoch vs Loss.jpg', dpi=500, bbox_inches='tight')
        plt.show()


########################################################################################################################

torch.manual_seed(2)                                                            # specify seed to get same values each time
model_perceptron = Perceptron(2, 1)                                             # Perceptron class object
print(model_perceptron.get_params())

# Create Dataset
n_points = 100
centers = [[-0.5, 0.5], [0.5, -0.5]]
sample_coordinates, sample_labels = datasets.make_blobs(n_samples=n_points, random_state=123, centers=centers, cluster_std=0.4)

# print(sample_coordinates)

# plot dataset and initial fit (with random weights and bias assigned by the seed)
model_perceptron.plot_fit_and_dataset('Initial Model', sample_coordinates, sample_labels)

coordinates_data = torch.Tensor(sample_coordinates)
labels_data = torch.Tensor(sample_labels.reshape([100, 1]))

# loss function will be measured based on binary cross entropy loss
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model_perceptron.parameters(), lr=0.01)
epochs = 2000                                                                   # number of iterations (may cause underfitting or overfitting)

# iterations
losses = []
for i in range(epochs):
    y_pred = model_perceptron.forward(coordinates_data)                         # make a prediction
    loss = criterion(y_pred, labels_data)
    losses.append(loss)

    if i % 50 == 0:
        print('epoch:', i, 'loss:', loss.item())

    optimizer.zero_grad()                                                       # zeros (resets grad), (grads accumulates after each backward)
    loss.backward()                                                             # derivative (gradient) of loss
    optimizer.step()                                                            # optimizer recalculates params (can be called after loss.backward())

model_perceptron.plot_fit_and_dataset('Trained Model', sample_coordinates, sample_labels)
model_perceptron.plot_loss_epoch(epochs, losses)


# Test Model
test_points = [torch.tensor([1.0, -1.0]), torch.tensor([-1.0, 1.0])]
for point in test_points:
    plt.plot(point.numpy()[0], point.numpy()[1], 'ko')

model_perceptron.plot_fit_and_dataset('Test Model', sample_coordinates, sample_labels)

print('\nTest:')
for point in test_points:
    print('Test point ', point.numpy(), ", class: ", model_perceptron.predict(point), ', probability: ',  model_perceptron.forward(point).item(), sep='')     # forward(point) shows the probability of a point belonging to a class

