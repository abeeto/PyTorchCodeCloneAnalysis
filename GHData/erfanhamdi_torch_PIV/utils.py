import numpy as np
import torch
from NCC import NCC

def torch_corr(c1: np.array,  sw: np.array) -> torch.Tensor:
    """
    Compute the normalized cross correlation between two images.
    
    :param c1: Interrogation window from the first image.
    :param sw: search window from the second image.

    :return: normalized cross correlation
    
    """
    sw = sw.unsqueeze(0)
    # instantiate the NCC class with the interrogation window
    ncc = NCC(c1)
    # compute the normalized cross correlation
    corr = ncc(sw)    
    return corr

def fixer(vecx: np.array, vecy: np.array, vec: np.array, rij:np.array,
         r_limit: float, i_fix: int) -> np.array:
    """
    Fixing the irregular vectors (Normalized Median Test and low Correlation coeff.)

    :param vecx: array containing the x displacement.
    :param vecy: array containing the y displacement.
    :param vec: array containing the total displacement.
    :param rij: array containing the correlation coefficient.
    :param r_limit: limit for the correlation coefficient.
    :param i_fix: Maximum number of iteration for fixing the irregularities.

    :return: array containing the fixed vectors.

    """
    fluc = np.zeros(vec.shape)
    for j in range(1, vec.shape[1] - 1):
        for i in range(1, vec.shape[0] - 1):
            neigh_x = np.array([])
            neigh_y = np.array([])
            for ii in range( -1, 2):
                for jj in range( -1, 2):
                    if ii == 0 and jj == 0: continue
                    neigh_x = np.append(neigh_x, vecx[i + ii, j + jj]) # Neighbourhood components
                    neigh_y = np.append(neigh_y, vecy[i + ii, j + jj])
            res_x = neigh_x - np.median(neigh_x) # Residual
            res_y = neigh_y - np.median(neigh_y)
            
            res_s_x = np.abs(vecx[i, j] - np.median(neigh_x)) / (np.median(np.abs(res_x)) + 0.1) # Normalized Residual (Epsilon=0.1)
            res_s_y = np.abs(vecy[i, j] - np.median(neigh_y)) / (np.median(np.abs(res_y)) + 0.1)
            
            fluc[i, j]=np.sqrt(res_s_x * res_s_x + res_s_y * res_s_y) # Normalized Fluctuations
    
    i_disorder = 0
    for ii in range(i_fix): # Correction Cycle for patches of bad data
        i_disorder = 0
        vec_diff = 0.0
        for j in range(1, vec.shape[1] - 1):
            for i in range(1, vec.shape[0] - 1):
                if fluc[i, j] > 2.0 or (rij[i, j] < r_limit): # Fluctuation threshold = 2.0
                    i_disorder += 1
                    vecx[i, j] = 0.25 * (vecx[i + 1, j] + vecx[i - 1, j] + vecx[i, j + 1] + vecx[i, j - 1]) # Bilinear Fix
                    vecy[i, j] = 0.25 * (vecy[i + 1, j] + vecy[i - 1, j] + vecy[i, j + 1] + vecy[i, j - 1])
                    vec_diff += (vec[i, j] - np.sqrt(vecx[i, j] * vecx[i, j] + vecy[i, j] * vecy[i, j])) ** 2.0
                    vec[i, j] = np.sqrt(vecx[i, j] * vecx[i, j] + vecy[i, j] * vecy[i, j])
                    
        if i_disorder == 0 or vec.mean() == 0.0: break # No need for correction
        correction_residual = vec_diff / (i_disorder * np.abs(vec.mean()))
        if correction_residual < 1.0e-20: break # Converged!
    if ii == i_fix - 1: print("Maximum correction iteration was reached!")
    return vecx, vecy, vec, i_disorder, ii

def subpix(R: np.array, axis: str,  dum: np.array) -> float: 
    """
    Subpixle resolution (parabolic - Gaussian fit)

    :param R: array containing the correlation coefficient.
    :param axis: axis to be used for the subpixel resolution.
    :param dum: dummy array containing the coordinates of the max_corr in the R.
    
    :return: float as the subpixel resolution.

    """
    R_x = dum[1]
    R_y = dum[2]
    r = R[R_x, R_y]
    if np.abs(r - 1.0) < 0.01: return 0.0
    try: # Out of bound at the edges:
        if axis == 'y': #For vecy
            r_e = R[R_x + 1, R_y]
            r_w = R[R_x - 1, R_y]
        else:          #For Vecx
            r_e = R[R_x, R_y + 1]
            r_w = R[R_x, R_y - 1]
        if r_e > 0.0 and r_w > 0.0 and r > 0.0: # Gaussian if possible (resolves pick locking)
            r_e = np.log(r_e)
            r_w = np.log(r_w)
            r = np.log(r)
        if (r_e + r_w - 2 * r) != 0.0:
            if np.abs((r_w - r_e) / (2.0 * (r_e + r_w - 2 * r))) < 1.0 and np.abs(r_e + 1) > 0.01 and np.abs(r_w + 1) > 0.01:
                return (r_w - r_e) / (2.0 * (r_e + r_w - 2 * r))
        return 0.0
    except:
        return 0.0