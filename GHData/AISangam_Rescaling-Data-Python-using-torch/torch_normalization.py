import torch
X = torch.tensor(([4, 877], [8, 666], [3, 6]), dtype=torch.float) # 3 X 2 tensor
print("Tensor X:",X)
print()
print("Shape of tensor to be normalized:",X.shape)
#0 means calculating the maximum value from the array column wise
X_max, _ = torch.max(X, 0)
#print the maximum value from each column
print()
print("Maximum value from each column of tensor X:",X_max)
#divide each column of x with the x_max
X = torch.div(X, X_max)
print()
print("Normalized X is:",X)
