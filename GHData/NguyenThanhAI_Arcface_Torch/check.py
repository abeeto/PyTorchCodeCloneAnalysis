import timeit

setup = '''
import numpy as np
a = np.arange(64).reshape(8, 8)
b = np.arange(64).reshape(8, 8)
''' 
test = '''
for i in range(1000):
    a @ b
'''
test1 = '''
for i in range(1000):
    np.matmul(a,b)
'''
test2 = '''
for i in range(1000):
    a.dot(b)
'''

'''print( timeit.timeit(test, setup, number=100) )
print( timeit.timeit(test1, setup, number=100) )
print( timeit.timeit(test2, setup, number=100) )'''

test1 = '''x=np.array([[2, 1, 3], [-2, 1, -3]])
w=np.array([1, 1, 1])
y=np.array([0, 1])
grad = (2 * np.linalg.multi_dot([w, x.T, x]) - 2 * np.matmul(y, x))/x.shape[0]
'''

test2 = '''x=np.array([[2, 1, 3], [-2, 1, -3]])
w=np.array([1, 1, 1])
y=np.array([0, 1])
grad = (2 * np.matmul(w, np.matmul(x.T, x)) - 2 * np.matmul(y, x))/x.shape[0]
'''

print( timeit.timeit(test1, setup, number=10000) )
print( timeit.timeit(test2, setup, number=10000) )