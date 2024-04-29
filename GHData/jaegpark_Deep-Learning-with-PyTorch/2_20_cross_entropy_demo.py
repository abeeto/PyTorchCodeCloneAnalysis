import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    ans = 0
    for i in range(len(Y)):
        temp = Y[i]*np.log(P[i]) + (1-Y[i])*np.log(1-P[i])
        ans -= temp
    return ans
