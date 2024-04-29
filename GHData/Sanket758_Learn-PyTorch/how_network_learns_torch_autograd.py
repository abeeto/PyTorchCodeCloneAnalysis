"""
Description: Implementing function for training loop in pytorch to understand how machine learning works 
            under the hood but this time using pytorch for backward pass

There are 3 major steps in learning:
    1. Forward pass - Here inputs and weights are sent through the network to get a prediction. prediction can be incorrect at the start but 
                    it will improve eventually as network learns.
    2. Calculate local gradients - Here network will calculate gradient at each node of the network according to its input and output
    3. Backward pass - Here we will use 'chain rule' to calculate overall loss for the whole network from local gradients
    4. Update weights - Optimizer step - We take one step of the optimizer in order to optimize the loss

We will take an example of linear regression and for our optimizer we will use gradient descent
"""

import torch
# Linear regression: y= w.x + b 
# for now lets ignore the bias term and focus on y = w.x

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

# !Important for updating weights, we need to tell pytorch to computer graph for this varaible, that's why requires_gard=True
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)  # initialize our weights as zero at the start

# Lets implement a forward function, which will make prediction, which takes input 'x'
def forward(x):
    return w*x  # here we will return y=w.x as we are using a linear function

# Lets calculate our loss after prediction
# for out example we are using MSE - mean squared error as out loss function
def loss_fn(y, y_hat):
    return ((y-y_hat)**2).mean()  # y-y_hat will give us loss then we square it and take mean


""" We will replace this manual gradient calculation with pytorch's autograd """
# Lets calculate gradients for our optimzer step
# MSE = 1/N * (w*x - y)**2  -- derivate of constant * f(x) = constant * derivative(x)
# dJ/dW = 1/N * 2x * (w*x - y) - w*x=y_hat
# def gradient(x, y, y_hat):
#     return np.dot(2*x, y_hat-y).mean()


# Now lets implement the training loop, but before let's predict using an dummy input to see the difference
print(f"Prediction before training: f(5) = {forward(5):.3f}, weights: {w:.3f}")

epochs = 20
learning_rate = 0.01    

print('+'*20, 'training', '+'*20)
for epoch in range(epochs):
    # step 1. Do the forward pass
    y_hat = forward(X)

    # Calculate loss
    loss = loss_fn(y, y_hat)

    # calculate gradients
    # dW = gradient(X, y, y_hat)
    loss.backward() # backward pass with autograd

    # take optimization step - for this we will use gradient descent
    # w -= learning_rate * dW  # learn more here: https://builtin.com/data-science/gradient-descent
    with torch.no_grad():
        w -= learning_rate * w.grad

    # As gradient get accumulated everytime, we need to make sure we make them zero before next epoch
    w.grad.zero_()

    if epoch%2==0:
        print(f"Epoch {epoch+1}: w={w:.3f} loss={loss:.10f}")

print('+'*20, 'done', '+'*20)

print(f'weights after training: {w:.3f}, lets do a prediction.')
print(f"Prediction after training: f(5) = {forward(5):.3f}")

"""
Output:
    Prediction before training: f(5) = 0.000, weights: 0.000
    ++++++++++++++++++++ training ++++++++++++++++++++
    Epoch 1: w=0.300 loss=30.0000000000
    Epoch 3: w=0.772 loss=15.6601877213
    Epoch 5: w=1.113 loss=8.1747169495
    Epoch 7: w=1.359 loss=4.2672529221
    Epoch 9: w=1.537 loss=2.2275321484
    Epoch 11: w=1.665 loss=1.1627856493
    Epoch 13: w=1.758 loss=0.6069811583
    Epoch 15: w=1.825 loss=0.3168478012
    Epoch 17: w=1.874 loss=0.1653965265
    Epoch 19: w=1.909 loss=0.0863380581
    ++++++++++++++++++++ done ++++++++++++++++++++
    weights after training: 1.922, lets do a prediction.
    Prediction after training: f(5) = 9.612

Here, you can see the with pytorch's autograd, we was not able to get correct output even with same number of epochs,
that is due to difference between internal numerical implementation of autograd package. But if you increase the epoch,
model does converge.
"""
