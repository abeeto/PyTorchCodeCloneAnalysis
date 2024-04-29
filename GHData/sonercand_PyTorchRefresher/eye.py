import numpy as np

a = np.ones(2)
b = np.array([2, 2])
c = a * b
print(c)
d = np.dot(a, b)
print(d)
print(b.shape)
b = b.reshape((2, 1))
print(b.shape)
print(b[1])
