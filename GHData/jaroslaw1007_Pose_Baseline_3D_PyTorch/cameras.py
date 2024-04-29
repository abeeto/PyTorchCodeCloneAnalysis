import numpy as np
import h5py
import itertools

from config import *

def project_point_radial(P, R, T, f, c, k, p):
    N = P.shape[0]
    X = R.dot(P.T - T)
    XX = X[:2, :] / X[2, :]
    
    r2 = XX[0, :]**2 + XX[1, :]**2
    
    radial = 1 + np.einsum('ij,ij->j', np.tile(k, (1, N)), np.array([r2, r2**2, r2**3]))
    tan = p[0] * XX[1, :] + p[1] * XX[0, :]
    
    XXX = XX * np.tile(radial + tan, (2, 1)) + np.outer(np.array([p[1], p[0]]).reshape(-1), r2)
    
    Proj = (f * XXX) + c
    Proj = Proj.T
    
    D = X[2, ]
    
    return Proj, D, radial, tan, r2

def load_camera_parameter(hf, path):
    R = hf[path.format('R')][:]
    R = R.T
    
    T = hf[path.format('T')][:]
    f = hf[path.format('f')][:]
    c = hf[path.format('c')][:]
    k = hf[path.format('k')][:]
    p = hf[path.format('p')][:]
    
    name = hf[path.format('Name')][:]
    name = ''.join(chr(item) for item in name)
    
    return R, T, f, c, k, p, name

def load_cameras(base_path='cameras.h5', subjects=[1, 5, 6, 7, 8, 9, 11], n_interpolations=0):
    rcams = {}
    
    with h5py.File(base_path, 'r') as hf:
        for s in subjects:
            for c in range(4):
                rcams[(s, (c + 1))] = load_camera_parameter(hf, 'subject%d/camera%d/{0}' % (s, c + 1))
    
    return rcams

def world_to_camera_frame(P, R, T):
    X_cam = R.dot(P.T - T)
    
    return X_cam.T
    
def camera_to_world_frame(P, R, T):
    X_cam = R.T.dot(P.T) + T
    
    return X_cam.T