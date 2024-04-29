'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''


class SERLU(nn.Module):
    def __init__(self):
        super(SERLU, self).__init__()
        self.alpha_serlu = 2.90427
        self.lambda_serlu = 1.07862
        
    def forward(self, x):
        '''
        Create half of the piece-wise function by making the appropriate values
        zero or alpha*e^x. Create the other half by making all positions where
        there isn't a zero, a zero, and all position where there are, a one.
        
        Adding these together yields ones or alpha*e^x, i.e. our multiplier if
        x>=0 or x<0, respectively.
        '''
        
        multiplier_side_1 = (x<0).type(torch.cuda.FloatTensor)*self.alpha_serlu*torch.exp(x)
        multiplier_side_2 = (multiplier_side_1!=0).type(torch.cuda.FloatTensor)*torch.ones(multiplier_side_1.shape).to(device)
        
        multiplier = multiplier_side_1 + multiplier_side_2
        output = self.lambda_serlu*multiplier*x
        
        return output



class Maxout(nn.Module):
    def __init__(self, pool_size):
        super(Maxout, self).__init__()
        self._pool_size = pool_size

    def forward(self, x):
        # Group neurons together by pool size, and take max of each pool.
        maxout = x.view(x.shape[0], x.shape[1]//self._pool_size, self._pool_size).max(2)
        return maxout[0]



class Channelout(nn.Module):
    '''
    Channel-out activation as described in the original paper by Wang and JaJa.
    "Winning" channel determined by max if arg_abs_max is False. If True,
    it is determined by the argmax of the absolute max.
    '''
    def __init__(self, pool_size, arg_abs_max=False):
        super(Channelout, self).__init__()
        self._pool_size = pool_size
        self.arg_abs_max = arg_abs_max

    def forward(self, x):
        # Group neurons together by pool size.
        raw_vals = x.view(x.shape[0], x.shape[1]//self._pool_size, self._pool_size).to(device)
        
        if self.arg_abs_max:
            # First get indices of absolute max values.
            mask_inds = torch.abs(raw_vals).max(2)[1]
            
            # Then obtain absolute max values, without losing the sign.
            max_vals = raw_vals.max(2)[0]
            min_vals = raw_vals.min(2)[0]
            
            mask_vals = torch.where((min_vals*(-1))>max_vals,min_vals,max_vals)
        else:
            # Take max of each pool, and the index of the max.
            mask_vals, mask_inds = raw_vals.max(2)
        
        # Force both tensors with max information in to the right shape.
        mask_shape = (x.shape[0],x.shape[1]//self._pool_size,1)
        mask_vals, mask_inds = mask_vals.view(*mask_shape).to(device), mask_inds.view(*mask_shape).to(device)
        
        # Scatter the max values into the max indices, leaving all other indices zero.
        mask = torch.zeros(raw_vals.shape).to(device).scatter_(2,mask_inds, mask_vals)
        
        # Put the resulting tensor back into the right shape for the next layer.
        channelout = mask.view(x.shape[0],x.shape[1]).to(device)
        
        return channelout








def selectActivation(function_name):
    if function_name=='selu':
        activation = nn.SELU()
    elif function_name=='serlu':
        activation = SERLU()
    elif function_name=='elu':
        activation = nn.ELU()
    elif function_name=='leaky_relu':
        activation = nn.LeakyReLU()
    elif function_name=='relu':
        activation = nn.ReLU()
    else:
        activation = nn.Linear()
        print ('Defaulting to linear activation.')
    
    return activation

def createActivationLayers(activation_type, num_layers, pool_size=2):
    # Given a list of pool sizes, make each layer accordingly.
    # Given a single value, make all layers based on that value.
    if type(pool_size) is list:
        activation_params = pool_size
    else:
        activation_params = np.repeat(pool_size, num_layers)
    
    # Create layers based on given type and pool sizes.
    if activation_type=='selu':
        act_layers = nn.ModuleList([nn.SELU() \
            for a_p in activation_params])
    elif activation_type=='serlu':
        act_layers = nn.ModuleList([SERLU() \
            for a_p in activation_params])
    elif activation_type=='elu':
        act_layers = nn.ModuleList([nn.ELU() \
            for a_p in activation_params])
    elif activation_type=='leaky_relu':
        act_layers = nn.ModuleList([nn.LeakyReLU() \
            for a_p in activation_params])
    elif activation_type=='relu':
        act_layers = nn.ModuleList([nn.ReLU() \
            for a_p in activation_params])
    elif activation_type=='maxout':
        act_layers = nn.ModuleList([Maxout(a_p) \
            for a_p in activation_params])
    elif activation_type=='channelout':
        act_layers = nn.ModuleList([Channelout(a_p) \
            for a_p in activation_params])
    elif activation_type=='channelout_absmax':
        act_layers = nn.ModuleList([Channelout(a_p,True) \
            for a_p in activation_params])
    else:
        print ('Defaulting to relu activation.')
        act_layers = nn.ModuleList([nn.ReLU() \
            for a_p in activation_params])
    
    return act_layers
