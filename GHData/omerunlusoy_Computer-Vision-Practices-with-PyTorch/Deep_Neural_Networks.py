import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets                                                         # to create linearly separable data set

""" Deep Neural Networks """
# Deep (Learning) Neural Network includes more than 1 hidden layer
# Feedforward Process: Receiving some input, producing some output, and making prediction (no feedback loops and connections)
# Depth = number of hidden layer
# deeper neural networks can learn more complex functions

# https://playground.tensorflow.org/


########################################################################################################################

""" Deep Neural Network Class """
class DNN(nn.Module):

    def __init__(self, input_size, H1, output_size):
        super().__init__()
        self.linear1 = nn.Linear(in_features=input_size, out_features=H1)
        self.linear2 = nn.Linear(in_features=H1, out_features=output_size)


    def forward(self, x):
        x = torch.sigmoid(self.linear1(x.float()))
        x = torch.sigmoid(self.linear2(x))
        return x


    def predict(self, x):
        prediction = self.forward(x).item()
        if prediction >= 0.5:
            return 1
        else:
            return 0


    def get_params(self):
        print(self.parameters())
        [w, b] = self.parameters()                                                  # returns the weights and bias of given model
        weight1 = w[0][0].item()
        weight2 = w[0][1].item()
        bias = b[0].item()
        return weight1, weight2, bias


    def model_size(self):
        n_params = 0
        for param in self.parameters():
            n_params += param.numel()
        return n_params


    def plot_decision_boundary(self, coordinates, labels):
        x_span = np.linspace(min(coordinates[:, 0]) - 0.25, max(coordinates[:, 0]) + 0.25)
        y_span = np.linspace(min(coordinates[:, 1]) - 0.25, max(coordinates[:, 1]) + 0.25)
        xx, yy = np.meshgrid(x_span, y_span)

        grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()])                          # ravel is just to flatten the arrays
        pred_func = self.forward(grid)

        z = pred_func.view(xx.shape).detach().numpy()                               # reshapes the pred_func to xx
        plt.contourf(xx, yy, z)

        # plot dataset
        plt.scatter(coordinates[labels == 0, 0], coordinates[labels == 0, 1])       # first center samples
        plt.scatter(coordinates[labels == 1, 0], coordinates[labels == 1, 1])       # second center samples

        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.grid(color='gray', linestyle='-', linewidth=0.2)
        # plt.savefig('Decision Boundary.jpg', dpi=500, bbox_inches='tight')
        plt.show()


    def plot_loss_epoch(self, epochs, losses):
        plt.title('Epoch vs Loss')
        plt.plot(range(epochs), losses)
        plt.xlabel('epoch')
        plt.ylabel('Loss')
        plt.grid(color='gray', linestyle='-', linewidth=0.2)
        # plt.savefig('Epoch vs Loss.jpg', dpi=500, bbox_inches='tight')
        plt.show()


# Variables ############################################################################################################

seed = 2                        # seed

input_size = 2                  # input layer node number
H1 = 4                          # hidden layer node number (1 hidden layer)
output_size = 1                 # output layer node number

n_points = 500                  # data points
random_state = 123
noise = 0.1                     # noise for both circles
factor = 0.2                    # circles' relative sizes, 1 means same circles

learning_rate = 0.01
epochs = 1000

print_epoch_interval = 50

########################################################################################################################

torch.manual_seed(seed)                                                                             # specify seed to get same values each time

# this model uses 4 nodes in its hidden layer; although there is no optimal number,
# too little nodes reduce accuracy and too many nodes cause overfitting.
model_DNN = DNN(input_size, H1, output_size)                                                        # Perceptron class object

# Create Dataset
sample_coordinates, sample_labels = datasets.make_circles(n_samples=n_points, random_state=random_state, noise=noise, factor=factor)  # factor: circle relative sizes, 1 means same circle

# print(sample_coordinates)

# plot dataset and initial fit (with random weights and bias assigned by the seed)
# model_DNN.plot_fit_and_dataset('Initial Model', sample_coordinates, sample_labels)
model_DNN.plot_decision_boundary(sample_coordinates, sample_labels)

coordinates_data = torch.Tensor(sample_coordinates)
labels_data = torch.Tensor(sample_labels.reshape([n_points, 1]))


# Adam Optimization Algorithm (recommended default algorithm for optimization)
# combination of two extensions of Stochastic Gradient Descent (Adagrad and RMSprop)
# computes adaptive learning rates for each parameter (most important feature)

# loss function will be measured based on Binary Cross Entropy Loss
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model_DNN.parameters(), lr=learning_rate)

# iterations
losses = []
for i in range(epochs):
    y_pred = model_DNN.forward(coordinates_data)                                # make a prediction
    loss = criterion(y_pred, labels_data)
    losses.append(loss)

    if i % print_epoch_interval == 0:
        print('epoch:', i, 'loss:', loss.item())

    optimizer.zero_grad()                                                       # zeros (resets grad), (grads accumulates after each backward)
    loss.backward()                                                             # derivative (gradient) of loss
    optimizer.step()                                                            # optimizer recalculates params (can be called after loss.backward())


model_DNN.plot_loss_epoch(epochs, losses)


# Test Model
test_points = [torch.tensor([0.1, -0.1]), torch.tensor([-1.0, 1.0])]
for point in test_points:
    plt.plot(point.numpy()[0], point.numpy()[1], 'ro')

model_DNN.plot_decision_boundary(sample_coordinates, sample_labels)

print('\nTest:')
for point in test_points:
    print('Test point ', point.numpy(), ", class: ", model_DNN.predict(point), ', probability: ',  model_DNN.forward(point).item(), sep='')     # forward(point) shows the probability of a point belonging to a class
