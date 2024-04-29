# -*- coding: utf-8 -*-
"""
Created on August 2022

Neural network architecture.

@authors: Nuria Gómez-Vargas
"""

import torch

##########################################################################################

# SEED

seed = 16
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

##########################################################################################


class NN(torch.nn.Module):
    """
    Neural network that takes as input the already computed elementwise l1-distance between
    the pair of features and gives the probability of similarity.
    
    Parameters
    ----------
        output_layer: torch.nn.Linear
            Computation of the weighted l1 distance.
        
        slope: float
            Slope for the Gaussian activation function.
    """
    
    def __init__(self, input_size, slope = -0.05):
        """
        It instantiates the neural network.
    
        Parameters
        ----------
            input_size: int
                Size of the vector of features.
        
            slope: float
                Slope for the Gaussian activation function.
        """
        
        super(NN, self).__init__()
        
        # Definimos capas
        self.output_layer = torch.nn.Linear(in_features = input_size, out_features = 1, bias = True) # el bias=True se lo pone a la capa anterior
        self.slope = slope

    # Dentro de las clases, se pueden crear funciones que aplican a la clase definida.
    # Esta función forward la utilizamos para computar la pasada hacia adelante, 
    # que en este caso es la predicción realizada por la red.
    def forward(self, x):
        """
        This function computes the forward step on input x.

        Parameters
        ----------
        x: Tensor
            Input of the network.

        Returns
        -------
        o: Tensor
            Output of the network. Probability of similarity through the Gaussian activation function.

        """
        
        o = self.output_layer(x)
        o = torch.exp(  torch.mul(self.slope, torch.square(o)) ) #Después de cada multiplicación de matrices al pasar de una capa a otra, existe una función de activación

        return o
