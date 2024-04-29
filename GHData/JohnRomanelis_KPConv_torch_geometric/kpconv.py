from torch_geometric.nn import MessagePassing
from kpconv_utils import *

import torch.nn as nn

class KPConvRigit(MessagePassing):
    '''
       numKernels : number of kernels
       sigma : sigma parameter of the kernels
       in_channels : number of input features
       out_channels : number of output features
       device :  the device where the model is trainned on
       normalization_factor : initial normalization of the weight matrix (following a normal destribution) # this is a temporary fix
                                                                                                           # will use a init_function 
                                                                                                           # if i make the kernel work
    '''

    def __init__(self, numKernels, sigma, in_channels, out_channels, device = "cuda:0", normalization_factor = 0.1):
        super(KPConvRigitStrided, self).__init__("add")

        self.numKernels = numKernels
        self.sigma = sigma#torch.tensor(sigma, device=device) # turning sigma to a tensor for gpu opperations
        self.device = device
        # if in_channels == 0, we add an artificial channel containing ones
        if in_channels == 0:
            self.in_channels = 1
        else:
            self.in_channels = in_channels

        self.out_channels = out_channels

        # creating the kernels
        # self.kernels has shape : (numKernels) x 3
        self.kernels = initializeRigitKernels(numKernels).to(device)

        # creating per kernel weight matrix
        # self.weights has shape : (numKernels) x (in_channels) x (out_channels)
        self.weights = nn.Parameter(torch.randn((self.numKernels, self.in_channels, self.out_channels)))
        #self.weights*= normalization_factor
        


    def forward(self, pos, assigned_index, N, M, h = None ):
        '''
        - pos : node position
        - assigned_index : connection of nodes
        - N : number of input points
        - M : number of subsampled points
        - h : node feature
        '''
        
        # start propagating messages
        return self.propagate(edge_index = assigned_index, h=h, pos=pos, size=(N,M))

    def message(self, h_j, pos_j, pos_i):
        '''
        - h_j defines the features of neighboring nodes as shape [num_edges, in_channels]
        - pos_j defines the position of neighboring nodes as shape [num_edges, 3]
        - pos_i defines the position of central nodes as shape [num_edges, 3]
        '''
        
        if h_j == None:
            h_j = torch.ones((pos_j.shape[0],1), device=self.device)

        ## Creating a matrix to save the result after passing it through all the kernels
        added_weights = torch.zeros((pos_j.shape[0],self.in_channels, self.out_channels), device = self.device)

        # passing the points through all kernel points to calculate the weight matrix
        for kernel in range(self.numKernels):

            added_weights += linearKernel(pos_j-pos_i, self.kernels[kernel][:], self.sigma).unsqueeze(-1).unsqueeze(-1) * self.weights[kernel][:][:]
        
        # unsqueezing to make the tensors have the appropriate size for multiplication
        # the squezzing again to remove signleton dimension
        return (h_j.unsqueeze(1).matmul(added_weights)).squeeze(1)
