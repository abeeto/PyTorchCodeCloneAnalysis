import numpy as np 
import os
import platform
import pickle


file_path = '_datasets/cifar-10-batches-py'



def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return pickle.load(f)
    elif version[0] == '3':
        return pickle.load(f, encoding="latin1")
    raise ValueError("Invalid python version: {}".format(version))
    
def load_CIFAR_batch(filename):
    """load single batch of cifar"""
    with open(filename, "rb") as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3072)
        Y = np.array(Y)
        return X, Y
    
def load_CIFAR10(ROOT):
    """load all of cifar"""
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT,'data_batch_%d' %(b))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT,'test_batch'))
    return Xtr, Ytr, Xte, Yte
    
def get_CIFAR10_data(path):
    X_train, Y_train, X_test, Y_test = load_CIFAR10(path)
    return X_train, Y_train, X_test, Y_test


x_train, y_train, x_test, y_test = get_CIFAR10_data(file_path)

def merge_channels(im):
    r = (im[:1024]).reshape(32,32)
    g = (im[1024:2048]).reshape(32,32)
    b = (im[2048:]).reshape(32,32)
    return np.dstack((r,g,b))


    #alternative to the above
#     img = np.zeros((32,32,3),'uint8')
#     print(r.shape)
#     img[...,0] = r
#     img[...,1] = g
#     img[...,2] = b
#     return img.astype('float32')

x_train_merge = np.array([merge_channels(i)/255.0 for i in x_train])
x_test_merge = np.array([merge_channels(i)/255.0 for i in x_test])

print("[INFO] Saving...")

print("Saving data")
with open("train_imgs.npz", "wb") as f:
	np.savez_compressed(f, data=x_train_merge)
with open("train_lbs.npz", "wb") as f:
	np.savez_compressed(f, data=y_train)
with open("test_imgs.npz", "wb") as f:
	np.savez_compressed(f, data=x_test_merge)
with open("test_lbs.npz", "wb") as f:
	np.savez_compressed(f, data=y_test)
print("[INFO] Done...")

