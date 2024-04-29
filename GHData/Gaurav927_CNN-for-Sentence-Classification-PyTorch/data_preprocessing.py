import numpy as np

X,X_test, label, label_test = train_test_split(data,label,test_size =0.1)

def iterator_func(X,y,batch_size=128):
    size = len(X)
    permutation = np.random.permutation(size)
    iterator = []
    iterator_y = []
    for i in range(0,size, batch_size):
        indices = permutation[i:i+batch_size]
        iterator.append(X[indices])
        iterator_y.append(y[indices])
    return iterator, iterator_y

