import numpy as np
import pandas as pd

data = pd.DataFrame(pd.read_csv("housing.csv"))

if bool(data.isnull) == True:
    data.dropna

# Normalise the Data
norm_data = (data-data.std())/(data.mean())

# Split Independent Variables (X-matrix) from Dependent Variables (Y-matrix)
X = norm_data.iloc[:,0:3].values
y = norm_data.iloc[:,3:4].values

# Add intercept term to X
intercept = np.ones((X.shape[0],1))
X = np.concatenate((intercept,X),axis=1)

# Generate Weights Matrix
theta = np.zeros([X.shape[1],1])

# Important Parameters
learning_rate = 0.001
num_iterations = 10000
training_examples = y.size
xT = np.transpose(X)


for iteration in range(num_iterations):
    hypothesis = np.dot(X,theta)
    
    loss = np.sum(hypothesis-y)
    cost = (loss**2)/(2*training_examples)

    gradient = np.dot(xT,(hypothesis-y))/(y.size)

    theta = theta - learning_rate * gradient
    print("Loss is:",loss)
	


print("Theta is:",theta)
print("Loss is:",loss)

