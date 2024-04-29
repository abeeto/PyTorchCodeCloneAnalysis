import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    soft_max = []
    denom = np.sum(np.exp(L))
    for i in range(len(L)):
        soft_max.append(np.exp(L[i])/denom)
    return soft_max
