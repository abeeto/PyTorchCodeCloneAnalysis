#%%
import torch
import matplotlib.pyplot as plt

x = torch.ones(2, 2, requires_grad=False)
print(x)

#%%

y = x + 3
print( y.requires_grad )
print( y )
# %%
##y was created as a result of an operation, so it has a grad_fn.
print(y.grad_fn)
# %%

z = y*2
z.requires_grad = True#
out = z.mean()
print(out)
# %%
out.backward()
# %%
print(x.grad)

# %%



################################################################################
########################## regression example ##################################
################################################################################

## 1. Data Generation ##########################################################

np.random.seed(42)
x = np.random.rand(100, 1)
y = 1 + 2 * x + .1 * np.random.randn(100, 1)

# Shuffles the indices
idx = np.arange(100)
np.random.shuffle(idx)

# Uses first 80 random indices for train
train_idx = idx[:80]
# Uses the remaining indices for validation
val_idx = idx[80:]

# Generates train and validation sets
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

plt.scatter(x ,y )

#%%
## 2. Numpy method #############################################################


# Initializes parameters "a" and "b" randomly
np.random.seed(42)
a = np.random.randn(1)
b = np.random.randn(1)

print(a, b)

# Sets learning rate
lr = 1e-1
# Defines number of epochs
n_epochs = 1000

for epoch in range(n_epochs):
    # Computes our model's predicted output
    yhat = a + b * x_train
    
    # How wrong is our model? That's the error! 
    error = (y_train - yhat)
    # It is a regression, so it computes mean squared error (MSE)
    loss = (error ** 2).mean()
    
    # Computes gradients for both "a" and "b" parameters
    a_grad = -2 * error.mean()
    b_grad = -2 * (x_train * error).mean()
    
    # Updates parameters using gradients and the learning rate
    a = a - lr * a_grad
    b = b - lr * b_grad
    
print(a, b)

# %%
## 2. pytorch method #############################################################

## Our data was in Numpy arrays, but we need to transform them into PyTorch's Tensors
# and then we send them to the chosen device

x_train_tensor = torch.from_numpy( x_train ).float()
y_train_tensor = torch.from_numpy( y_train ).float()
# %%

lr = 1e-1
n_epochs = 1000

torch.manual_seed(42)
a = torch.randn(1, requires_grad=True, dtype=torch.float )
b = torch.randn(1, requires_grad=True, dtype=torch.float )

for epoch in range(n_epochs):
    yhat = a + b * x_train_tensor
    error = y_train_tensor - yhat
    loss = (error ** 2).mean()

    # We just tell PyTorch to work its way BACKWARDS from the specified loss!
    loss.backward()

    with torch.no_grad():
        a -= lr * a.grad
        b -= lr * b.grad
    
    # PyTorch is "clingy" to its computed gradients, we need to tell it to let it go...
    a.grad.zero_()
    b.grad.zero_()
#%%
print( 'pytorch says ')
print('a = ' , a )
print('b = ' ,  b )
# %%
