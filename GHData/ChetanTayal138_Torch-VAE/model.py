from torch.autograd import Variable
from utils import latent_space
import torch


class Weights:

    def __init__(self, name, dimension, initialization=None):
        self.name = name
        self.dimension = dimension
        self.initialization = None
        self.weights = torch.empty(dimension, requires_grad=True)
        self.flag = 0
        
    
    def __repr__(self):
        return self.name

    def _get_weight(self):
        if self.flag == 0:
            self.weights = torch.nn.init.xavier_normal_(self.weights)
            self.flag = 1

        return self.weights

    def __mul__(self, other):
        return torch.matmul(other, self._get_weight())

    def __add__(self, other):
        return torch.add(self._get_weight(), other)

    def __rmul__(self, other):
        return torch.matmul(other, self._get_weight())

    def __radd__(self, other):
        return torch.add(self._get_weight(), other)


class Encoder:

    def __init__(self, input_image):
        self.input = Variable(input_image, requires_grad = True)
        
    def encode(self, weights, biases):
        encoder_layer = self.input * weights["w1"] + biases["b1"]
        encoder_layer = torch.tanh(encoder_layer)
        mean_layer = encoder_layer * weights["w2"] + biases["b2"]
        std_dev_layer = encoder_layer * weights["w3"] + biases["b3"]
        return mean_layer, std_dev_layer


class Decoder:
    
    def __init__(self, latent_space):
        
        if str(type(latent_space)) == "<class 'numpy.ndarray'>":
            latent_space = torch.from_numpy(latent_space).float()
            self.latent_space = Variable(latent_space, requires_grad = False)

        else:
            self.latent_space = Variable(latent_space, requires_grad = True)

    def decode(self, weights, biases):
        decoder_layer = self.latent_space * weights["w4"] + biases["b4"]
        decoder_layer = torch.tanh(decoder_layer)
        decoder_output = decoder_layer * weights["w5"] + biases["b5"]
        decoder_output = torch.sigmoid(decoder_output)

        return decoder_output

if __name__ == "__main__" :

    IMAGE_DIM = 28 
    NN_DIM = 10

    input_image = Variable(torch.randn(10,28))
    w1 = Weights("weight_matrix_encoder_hidden", [IMAGE_DIM, NN_DIM])
    print(input_image*w1)
    
    




