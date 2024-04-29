import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# compute the derivative of outputs associated with inputs
def grad(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)

""" Gradient """
# Instantaneous rate of change (Derivative)
x = torch.tensor(2.0, requires_grad=True)
t = torch.tensor(2.0, requires_grad=True)
y = 9*x**4 + 2*x**3 + 3*x**2 + 6*x + 1 + t
u = 9*x**4 + 2*x**3 + 3*x**2 + 6*x + 1
u.backward()
x_grad = x.grad
print('x_grad:', x_grad)                            # tensor(330.)

y_x = grad(y, x)[0]
print('y_x:', y_x)                                  # tensor(330.)
# u_t = y_x[:, [1]]
# print('u_t:', u_t)                                  # tensor(330.)


# Partial Derivatives
z = torch.tensor(2.0, requires_grad=True)
y = x**2 + z**3
y.backward()
x_grad = x.grad
z_grad = z.grad
# print('x_grad:', x_grad, 'z_grad:', z_grad)       # x_grad: tensor(4.) z_grad: tensor(12.)


########################################################################################################################

""" Linear Regression """
def forward(x, weight, bias):
    y = weight * x + bias
    return y


# MAIN
w = torch.tensor(3.0, requires_grad=True)           # random initialization of weight
b = torch.tensor(1.0, requires_grad=True)           # random initialization of bias

torch.manual_seed(1)                                # randomness (1 means deterministic)
model = nn.Linear(in_features=1, out_features=1)    # Linear Model (for every output, there is a single input)
# print(model.weight, model.bias)                   # tensor([[0.5153]], requires_grad=True), tensor([-0.4414], requires_grad=True)

x = torch.tensor([[2.0], [3.3]])
y = forward(x, model.weight, model.bias)
y_predict = model(x)
# print("y:", y, "\ny_predict:", y_predict)         # y:         tensor([[0.5891], [1.2590]], grad_fn=<AddBackward0>)
                                                    # y_predict: tensor([[0.5891], [1.2590]], grad_fn=<AddmmBackward>)

########################################################################################################################

""" Custom Modules """
class Linear_Regression(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=output_size)

    def forward(self, x):
        pred = self.linear(x)
        return pred

    def get_params(self):
        [w, b] = self.parameters()                 # returns the weight and bias of given model
        # print(w, b)                              # tensor([[0.5153]], requires_grad=True), tensor([-0.4414], requires_grad=True)
        weight = w[0][0].item()
        bias = b[0].item()
        return weight, bias

    def plot_fit_and_dataset(self, title, x, y):
        plt.title(title)
        w, b = self.get_params()
        x1 = np.array([-30, 30])
        y1 = w*x1 + b

        plt.plot(x1, y1, 'r')                       # fit
        plt.plot(x.detach().numpy(), y.detach().numpy(), 'o')         # dataset, matplotlib works best with numpy arrays -OR- plt.scatter(X, y)

        plt.xlabel('x')
        plt.ylabel('y')
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


torch.manual_seed(1)
custom_model = Linear_Regression(1, 1)
x = torch.tensor([[2.0], [3.3]])
y_predict = custom_model.forward(x)
# print("y_predict:", y_predict)                    # y_predict: tensor([[0.5891], [1.2590]], grad_fn=<AddmmBackward>)


########################################################################################################################

""" Creating Dataset """

X = torch.randn(100, 1) * 10                        # Normally distributed random values (#100, 1 column, 0 centered)
Y = X + torch.randn(100, 1)*3                       # add some noise to data

w, b = custom_model.get_params()
# print(w, b)                                       # 0.5152631998062134 -0.44137823581695557
custom_model.plot_fit_and_dataset('Initial Model', X, Y)


########################################################################################################################

""" Loss Function """
# Gradient Descent (Simple Optimization Algorithm), (minimizes the error function)
# LOSS = (y1 - y_pred)² = (y1 - w*x1)²
# moving negative of the gradient of error function takes us to the lowest level

""" Mean Squared Error (MSELoss) """
# MSE = (1/n) SUM[(y - y_pred)²] = (1/n) SUM[(y - (mx + b))²]

criterion = nn.MSELoss()                            # loss function (mean squareds error), (we want to minimize this)

# optimizes (updates) model's parameters
# stochastic gradient descent (more optimum than bath gradient descent)
# lr = learning rate, tiny steps at each iteration
optimizer = torch.optim.SGD(custom_model.parameters(), lr=0.001)

epochs = 100                                        # number of iterations (may cause underfitting or overfitting)

# iterations
losses = []
for i in range(epochs):
    y_pred = custom_model.forward(X)                # make a prediction
    loss = criterion(y_pred, Y)
    print('epoch:', i, 'loss:', loss.item())
    losses.append(loss)

    optimizer.zero_grad()                           # zeros (resets grad), (grads accumulates after each backward)
    loss.backward()                                 # derivative (gradient) of loss
    optimizer.step()                                # optimizer recalculates params (can be called after loss.backward())

custom_model.plot_loss_epoch(epochs, losses)
custom_model.plot_fit_and_dataset('Trained Model', X, Y)
