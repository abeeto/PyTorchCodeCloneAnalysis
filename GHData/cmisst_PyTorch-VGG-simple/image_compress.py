import numpy as np

# apply FFT and remove high frequency, inverse transform
# apply SVD and remove minor singular values, reconstruct

# based on CPU, no parallel/GPU optimization

class img_comp(object):

    def __init__(self, arg):
        self.image = None
        self.dimension = 0
        assert(isinstance(arg, (np.ndarray, np.array)))
        self.dimension = np.size(np.shape(arg))
        assert(self.dimension == 2 or self.dimension == 3)
        if self.dimension == 2:
            # greyscale image
            pass
        else:
            # RGB image
            raise NotImplementedError
            # assert(np.shape(arg)[2] == 3)
        self.image = arg
        # return super(img_comp, self).__init__()
    
    def linear_dependent(self, rank):
        if isinstance(rank, int):
            assert(rank<=np.min(np.shape(self.image)[0:2]))
        elif isinstance(rank, float):
            assert(rank>=0.0 and rank<=1.0)
            rank = int(rank*np.min(np.shape(self.image)[0:2]))
        else:
            raise ArithmeticError
        u, s, vh = np.linalg.svd(self.image, full_matrices=True)
        s[rank:] = 0
        reconstruct = (u * s) @ vh 
        reconstruct[reconstruct<=0]=0
        reconstruct[reconstruct>=255]=255
        return reconstruct.astype(np.int)

    def low_frequency_pass(self, radius):
        raise NotImplementedError




if __name__== "__main__":
    import torch
    import torchvision
    import matplotlib.pyplot as plt
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    for i in range(5):
        plt.imshow(trainset.data[i,:,:,:])
        plt.show()
        test = np.dstack((
            img_comp(trainset.data[i,:,:,0]).linear_dependent(10),
            img_comp(trainset.data[i,:,:,1]).linear_dependent(10),
            img_comp(trainset.data[i,:,:,2]).linear_dependent(10)))
        plt.imshow(test)
        plt.show()
        