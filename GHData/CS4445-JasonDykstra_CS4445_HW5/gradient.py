from problem1 import *
import numpy as np

#-----------------------------------------------------------------
# gradient checking 
#-----------------------------------------------------------------

#--------------------------
def check_dz_db(x,w, b, delta=1e-7):
    '''
        compute the partial gradients of the logit function z w.r.t. the bias b using gradient checking.
        The idea is to add a small number to the weights and b separately, and approximate the true gradient using numerical gradient.
        For example, the true gradient of logit z w.r.t. bias can be approximated as  [z(w,b+ delta) - z(w,b)] / delta , here delta is a small number.
        Input:
            x: the feature vector of a data instance, a float numpy vector of length p. Here p is the number of features/dimensions.
            w: the weight vector of the logistic model, a float numpy vector of length p. 
            b: the bias value of the logistic model, a float scalar.
            delta: a small number for gradient check, a float scalar.
        Output:
            dz_dw: the approximated partial gradient of logit z w.r.t. the weight vector w computed using gradient check, a numpy float vector of length p. 
            dz_db: the approximated partial gradient of logit z w.r.t. the bias b using gradient check, a float scalar.
    '''
    dz_db = (compute_z(x, w, b+delta) - compute_z(x, w, b)) / delta
    return  dz_db

#--------------------------
def check_dz_dw(x,w, b, delta=1e-7):
    '''
        compute the partial gradients of the logit function z w.r.t. weights w using gradient checking.
        The idea is to add a small number to the weights and b separately, and approximate the true gradient using numerical gradient.
        Input:
            x: the feature vector of a data instance, a float numpy vector of length p. Here p is the number of features/dimensions.
            w: the weight vector of the logistic model, a float numpy vector of length p. 
            b: the bias value of the logistic model, a float scalar.
            delta: a small number for gradient check, a float scalar.
        Output:
            dz_dw: the approximated partial gradient of logit z w.r.t. the weight vector w computed using gradient check, a numpy float vector of length p. 
    '''
    p = x.shape[0] 
    dz_dw = np.zeros(p)
    for i in range(p):
        d = np.zeros(p)
        d[i] = delta
        dz_dw[i] = (compute_z(x,w+d, b) - compute_z(x, w, b)) / delta
    return dz_dw




#--------------------------
def check_dL_dz(z, y, delta=1e-7):
    '''
        Compute local gradient of the cross-entropy function w.r.t. the logits using gradient checking.
        Input:
            z: the linear logit, a float scalar
            y: the label of a training instance, an integer scalar value. The values can be 0 or 1.
            delta: a small number for gradient check, a float scalar.
        Output:
            dL_dz: the approximated local gradient of the loss function w.r.t. the activation, a float scalar value.
    '''
    dL_dz = (compute_L(z+delta,y) - compute_L(z-delta,y)) / 2./delta
    return dL_dz 

#--------------------------
def check_dL_db(x,y,w,b, delta=1e-7):
    '''
       Given an instance in the training data, compute the gradient of the bias b using gradient check.
        Input:
            x: the feature vector of a training instance, a float numpy vector of length p. Here p is the number of features/dimensions.
            y: the label of a training instance, an integer scalar value. The values can be 0 or 1.
            w: the weight vector, a float numpy vector of length p.
            b: the bias value, a float scalar.
            delta: a small number for gradient check, a float scalar.
        Output:
            dL_db: the approximated gradient of the loss function w.r.t. the bias, a float scalar. 
    '''
    a1 = forward(x,w,b+delta)
    a2 = forward(x,w,b)
    L1 = compute_L(a1,y)
    L2 = compute_L(a2,y)
    dL_db = (L1 - L2) / delta
    return dL_db

#--------------------------
def check_dL_dw(x,y,w,b, delta=1e-7):
    '''
       Given an instance in the training data, compute the gradient of the weights w using gradient check.
        Input:
            x: the feature vector of a training instance, a float numpy vector of length p. Here p is the number of features/dimensions.
            y: the label of a training instance, an integer scalar value. The values can be 0 or 1.
            w: the weight vector, a float numpy vector of length p.
            b: the bias value, a float scalar.
            delta: a small number for gradient check, a float scalar.
        Output:
            dL_dw: the approximated gradient of the loss function w.r.t. the weight vector, a numpy float vector of length p. 
    '''
    p = x.shape[0] # number of features
    dL_dw = np.zeros(p)
    for i in range(p):
        d = np.zeros(p)
        d[i] = delta
        a1 = forward(x,w+d,b)
        a2 = forward(x,w,b)
        L1 = compute_L(a1,y)
        L2 = compute_L(a2,y)
        dL_dw[i] = (L1 - L2) / delta
    return dL_dw


