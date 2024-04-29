import numpy as np
#import timeit # for time checking

"""Helper function to calculate symmetry function
Definition of symmetry function can be found in
Behler, J. The Journal of Chemical Physics, 134(7), 74106. (2011).
(http://doi.org/10.1063/1.3553717)

Summary of available function:

# calculate the cutoff function

# calculate the derivative of cutoff function

# calculate the G2 symmetry function

# calculate the derivative of G2 symmetry function
"""

def cutofffunc(cut_radi, dist):
    res = 0.5*(np.cos(np.pi*dist/cut_radi)+1)
    res[dist >= cut_radi] = 0.
    return res

def dcutofffunc(cut_radi, dist):
    res = -0.5*(np.pi/cut_radi)*np.sin(np.pi*dist/cut_radi) 
    res[dist >= cut_radi] = 0.
    return res
    
def symfunc2(coeffs, idxs, duo):
    dist = duo['dists'][idxs]
    cutoff = duo['cutfunc'][idxs]
    res = np.sum(cutoff * np.exp(-coeffs[0]*(dist-coeffs[1])**2))
    return res

def dsymfunc2(coeffs, idxs, duo, center, tot_atom_numbers):
    res = np.zeros([tot_atom_numbers, 3])
    dist = duo['dists'][idxs]
    cutoff = duo['cutfunc'][idxs]
    dcutoff = duo['dcutfunc'][idxs]
    tmp = np.exp(-coeffs[0]*(dist-coeffs[1])**2) * \
          (dcutoff - 2*coeffs[0]*(dist-coeffs[1]) * cutoff) * \
          duo['dists_deri'][idxs]

    for i in range(tot_atom_numbers):
        if i != center:
            res[i,:] += np.sum(tmp[duo['indices'][idxs] == i], axis=0)
    res[center,:] -= np.sum(tmp[duo['indices'][idxs] != center], axis=0)
    return res

def symfunc4(coeffs, idxs, trio):
    cutoff = trio['cutfunc'][idxs]
    cos_val = trio['cos'][idxs]
    dist_ssum = trio['dists_sqsum'][idxs]
    res = np.sum((2**(1-coeffs[1]))*(1 + coeffs[2] * cos_val)**coeffs[1] * \
          np.exp(-coeffs[0]*dist_ssum) * \
          np.prod(cutoff, axis=1, keepdims=True))
    return res

def dsymfunc4(coeffs, idxs, trio, center, tot_atom_numbers):
    res = np.zeros([tot_atom_numbers, 3])
    
    eta = coeffs[0]
    zeta = coeffs[1]
    lamda = coeffs[2] # lambda has another function
    cutoff = trio['cutfunc'][idxs]
    cos_val = trio['cos'][idxs]
    dist_ssum = trio['dists_sqsum'][idxs]
    dcutoff = trio['dcutfunc'][idxs]
    dcos_val = trio['dcos'][idxs]
    indices = trio['indices'][idxs]
    dist = trio['dists'][idxs]
    cos_term = 1 + lamda * cos_val

    tmp = (2**(1-zeta)) * cos_term**(zeta - 1) * np.exp(-eta*dist_ssum) * \
          trio['dists_deri'][idxs]
    tmp[:,0:3] *= np.expand_dims(cutoff[:,1]*cutoff[:,2] * \
                  (cos_term[:,0]*dcutoff[:,0] - 2*eta*dist[:,0]*cos_term[:,0]*cutoff[:,0] + \
                    cutoff[:,0]*zeta*lamda*dcos_val[:,0]), axis=1)
    tmp[:,3:6] *= np.expand_dims(cutoff[:,2]*cutoff[:,0] * \
                  (cos_term[:,0]*dcutoff[:,1] - 2*eta*dist[:,1]*cos_term[:,0]*cutoff[:,1] + \
                    cutoff[:,1]*zeta*lamda*dcos_val[:,1]), axis=1)
    tmp[:,6:9] *= np.expand_dims(cutoff[:,0]*cutoff[:,1] * \
                  (cos_term[:,0]*dcutoff[:,2] - 2*eta*dist[:,2]*cos_term[:,0]*cutoff[:,2] + \
                    cutoff[:,2]*zeta*lamda*dcos_val[:,2]), axis=1)

    for i in range(tot_atom_numbers):
        if i != center:
            idx_type1 = np.logical_and(indices[:,0] == i, indices[:,1] != i)
            idx_type2 = np.logical_and(indices[:,0] != i, indices[:,1] == i)
            idx_type3 = np.logical_and(indices[:,0] == i, indices[:,1] == i)
            res[i,:] += np.sum(tmp[idx_type1,0:3] - tmp[idx_type1,6:9], axis=0) + \
                        np.sum(tmp[idx_type2,3:6] + tmp[idx_type2,6:9], axis=0) + \
                        np.sum(tmp[idx_type3,0:3] + tmp[idx_type3,3:6], axis=0)

    idx_type1 = np.logical_and(indices[:,0] != center , indices[:,1] == center)
    idx_type2 = np.logical_and(indices[:,0] == center , indices[:,1] != center)
    idx_type3 = np.logical_and(indices[:,0] != center , indices[:,1] != center)
    res[center,:] += -np.sum(tmp[idx_type1,0:3] - tmp[idx_type1,6:9], axis=0) + \
                     -np.sum(tmp[idx_type2,3:6] + tmp[idx_type2,6:9], axis=0) + \
                     -np.sum(tmp[idx_type3,0:3] + tmp[idx_type3,3:6], axis=0)
    
    return res
