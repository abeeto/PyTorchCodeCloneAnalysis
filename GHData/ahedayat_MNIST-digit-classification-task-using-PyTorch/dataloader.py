import mnist
import numpy as np
from sklearn.decomposition import PCA

class DataLoader:
    def __init__(self,dataset_address,train_val_percent = 0.9 , pca_n_components = None, normalize = False):
        mnist.temporary_dir = lambda: dataset_address
        train_val_imgs = mnist.train_images()
        train_val_labels = mnist.train_labels()
        train_val_size = len(train_val_imgs)

        test_imgs = mnist.test_images()
        test_size = len(test_imgs)
        
        train_size = 50000

        self.pca_n_components = pca_n_components

        if pca_n_components is not None:
            pca = PCA(n_components=pca_n_components)

            train_val_imgs = train_val_imgs.reshape(train_val_size,-1)
            train_val_imgs = pca.fit(train_val_imgs).transform(train_val_imgs).reshape(train_val_size,-1,1)

            test_imgs = test_imgs.reshape(test_size,-1)
            test_imgs = pca.fit(test_imgs).transform(test_imgs).reshape(test_size,-1,1)

        if normalize:
            for ix in range(len(train_val_imgs)):
                train_val_imgs[ix] = (train_val_imgs[ix] - train_val_imgs[ix].mean()) / train_val_imgs[ix].std()
            for ix in range(len(test_imgs)):
                test_imgs[ix] = (test_imgs[ix] - test_imgs[ix].mean()) / test_imgs[ix].std()

        self.train_data = [ (np.reshape(img,(-1,1)),label) for img,label in zip(train_val_imgs[0:train_size],train_val_labels[:train_size])]
        self.val_data = [ (np.reshape(img,(-1,1)),label) for img,label in zip(train_val_imgs[train_size:],train_val_labels[train_size:])]
        self.test_data = [ (np.reshape(img,(-1,1)),label) for img,label in zip(test_imgs,mnist.test_labels())]


    def shape(self):
        if len(self.train_data)>0 :
            return self.train_data[0][0].shape
        return 0
    def num_classes(self):
        return 10
    # def size(self,train=True):
    #     if train:
    #         return len(self.train_images)
    #     return len(self.test_images)