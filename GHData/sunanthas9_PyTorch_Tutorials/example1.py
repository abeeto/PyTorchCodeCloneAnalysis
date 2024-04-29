# Network with single hidden layer, with 100 hidden units! Using only numpy and not pytorch, coding gradients by hand.

import numpy as np

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.

N, D_in, H, D_out = 64, 1000, 100, 10


# Create random input and output data

x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialize weights

w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H,D_out)

# Set learning rate

learning_rate = 1e-6

# Iterate through the loop 

for t in range(500):
	
	# Forward pass: compute predicted y
	h = x.dot(w1) # Multiply X with weights1 matrix
	h_relu = np.maximum(h,0) # Apply the ReLu function to calculate a1
	y_pred = h_relu.dot(w2) # Multiply a1 with weights2 

	# Compute and print loss
	loss = np.square(y_pred - y).sum() # Sum of squared error as cost
	print(t, loss) # Output iteration number with the cost
	
	# Backprop to compute gradients of w1 and w2 with respect to loss
	grad_y_pred = 2.0 * (y_pred-y) # differentiation of cost function
	grad_w2 = h_relu.T.dot(grad_y_pred) # Calculate intermediate matrix for differentiation
	grad_h_relu = grad_y_pred.dot(w2.T) # Recalculate based on new gradients
	grad_h = grad_h_relu.copy() # Make a copy of the new matrix
	grad_h[h <0] = 0 # Reapply ReLu function
	grad_w1 = x.T.dot(grad_h) # Calculate new weights1 gradient
	
	# Update weights
	w1 -= learning_rate * grad_w1
	w2 -= learning_rate * grad_w2	
