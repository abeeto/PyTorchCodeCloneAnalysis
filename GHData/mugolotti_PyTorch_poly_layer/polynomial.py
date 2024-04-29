import math,sys
import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
from torch.nn import Module


# http://home.iitk.ac.in/~shalab/regression/Chapter12-Regression-PolynomialRegression.pdf
# http://support.ptc.com/help/mathcad/en/index.html#page/PTC_Mathcad_Help/multivariate_polynomial_regression.html
# https://stats.stackexchange.com/questions/211468/interaction-term-in-multivariate-polynomial-regression

class Polynomial(torch.nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features', 'uni_deg', 'int_deg']

    def __init__(self, in_features, out_features, uni_deg, int_deg, bias=True):
        super(Polynomial, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.uni_deg = uni_deg
        self.int_deg = int_deg

        # Obtain number of terms in polynomial expansion
        deg_array = np.zeros((self.in_features,self.uni_deg+1))
        for ideg in range(0,self.uni_deg+1):
            deg_array[:,ideg] = ideg
        # All combination of degress in between variabels
        comb = np.array(np.meshgrid(*(deg_array[i,:]for i in range(self.in_features)))).T.reshape(-1,self.in_features)
        # Get all combination with degree less or equal to self.degree
        nweights = 0
        
        for icomb in range(0,np.size(comb,0)):
            # if all degrees are zero => fake poly
            if int(np.sum(comb[icomb,:])) == 0:
                comb[icomb,:] = None
            # Univariate polynomial and degree less than uni_deg
            elif np.count_nonzero(comb[icomb,:] != 0) == 1 and int(np.sum(comb[icomb,:])) <= self.uni_deg:
                nweights += 1
            # Univariate polynomial and degree greater than uni_deg
            elif np.count_nonzero(comb[icomb,:] != 0) == 1 and int(np.sum(comb[icomb,:])) >  self.uni_deg:
                comb[icomb,:] = None
            # Inrecorrelation: if all degrees are different than 0 and overall degree <= int_deg
            elif np.count_nonzero(comb[icomb,:] != 0) >  1 and int(np.sum(comb[icomb,:])) <= self.int_deg: 
                nweights += 1 
            # Inrecorrelation: if all degrees are different than 0 and overall degree >= int_deg
            elif np.count_nonzero(comb[icomb,:] != 0) >  1 and int(np.sum(comb[icomb,:])) >  self.int_deg: 
                comb[icomb,:] = None
            else:
                comb[icomb,:] = None
        
        # Save comb vector is allowed combination in class
        self.comb = comb

        self.weight = Parameter(torch.Tensor(out_features, nweights))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        
        # Stack basis
        # The catch here is that we can potential pass
        # only one set of inputs that is a vector.
        # In this case stacking in dim=1 is not possible.
        # We need to stack in dim=0
        if len(input.size()) == 1:
            # Convert vector in array
            input_tensor = input.unsqueeze(0)
            prediction = True
        else:
            input_tensor = input.clone()
            prediction = False
        
        # Creates polynomial basis
        poly_basis = []
        for icomb in range(0,np.size(self.comb,0)):
            if math.isnan(self.comb[icomb,0]): 
                pass
            else:
                term = torch.ones([input_tensor.size(0), 1], dtype=torch.float64)
                for ifeat in range(self.in_features):
                    term[:,0] *= input_tensor[:,ifeat].clone()**self.comb[icomb,ifeat]
                    #print term
                if prediction:
                    term = term[:,0]
                    #print term
                poly_basis.append(term)

        
        # Stack basis
        try:
            input_poly = torch.cat(poly_basis,dim=1)
        except:
            input_poly = torch.cat(poly_basis,dim=0)
    
        #print 'input_poly', input_poly
        return F.linear(input_poly, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, uni_deg{}, int_deg{}, bias={}'.format(
            self.in_features, self.out_features, self.uni_deg, self.int_deg, self.bias is not None
        )





