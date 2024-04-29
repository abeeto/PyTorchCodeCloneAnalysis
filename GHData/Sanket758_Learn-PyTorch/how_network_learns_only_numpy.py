"""
Description: Implementing manual function for training loop in pytorch to understand how machine learning works under the hood.
There are 3 major steps in learning:
    1. Forward pass - Here inputs and weights are sent through the network to get a prediction. prediction can be incorrect at the start but 
                    it will improve eventually as network learns.
    2. Calculate local gradients - Here network will calculate gradient at each node of the network according to its input and output
    3. Backward pass - Here we will use 'chain rule' to calculate overall loss for the whole network from local gradients
    4. Update weights - Optimizer step - We take one step of the optimizer in order to optimize the loss
We will take an example of linear regression and for our optimizer we will use gradient descent
We will only use numpy for this purpose.
"""
import numpy as np

# Linear regression: y= w.x + b 
# for now lets ignore the bias term and focus on y = w.x

X = np.array([1, 2, 3, 4], dtype=np.float32)  # X is our inputs
y = np.array([2, 4, 6, 8], dtype=np.float32)  # y is our output 

w = 0.0  # initialize our weights as zero at the start

# Lets implement a forward function, which will make prediction, which takes input 'x'
def forward(x):
    return w*x  # here we will return y=w.x as we are using a linear function

# Lets calculate our loss after prediction
# for out example we are using MSE - mean squared error as out loss function
def loss_fn(y, y_hat):
    return ((y-y_hat)**2).mean()  # y-y_hat will give us loss then we square it and take mean

# Lets calculate gradients for our optimzer step
# MSE = 1/N * (w*x - y)**2  -- derivate of constant * f(x) = constant * derivative(x)
# dJ/dW = 1/N * 2x * (w*x - y) - w*x=y_hat
def gradient(x, y, y_hat):
    return np.dot(2*x, y_hat-y).mean()

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
    dW = gradient(X, y, y_hat)

    # take optimization step - for this we will use gradient descent
    w -= learning_rate * dW  # learn more here: https://builtin.com/data-science/gradient-descent

    if epoch%2==0:
        print(f"Epoch {epoch+1}: w={w:.3f} loss={loss:.10f}")

print('+'*20, 'done', '+'*20)

print(f'weights after training: {w:.3f}, lets do a prediction.')
print(f"Prediction after training: f(5) = {forward(5):.3f}")


"""
Output: 
    Prediction before training: f(5) = 0.000, weights: 0.000
    ++++++++++++++++++++ training ++++++++++++++++++++
    Epoch 1: w=1.200 loss=30.0000000000
    Epoch 3: w=1.872 loss=0.7680001855
    Epoch 5: w=1.980 loss=0.0196608342
    Epoch 7: w=1.997 loss=0.0005033241
    Epoch 9: w=1.999 loss=0.0000128844
    Epoch 11: w=2.000 loss=0.0000003297
    Epoch 13: w=2.000 loss=0.0000000084
    Epoch 15: w=2.000 loss=0.0000000002
    Epoch 17: w=2.000 loss=0.0000000000
    Epoch 19: w=2.000 loss=0.0000000000
    ++++++++++++++++++++ done ++++++++++++++++++++
    weights after training: 2.000, lets do a prediction.
    Prediction after training: f(5) = 10.000
"""