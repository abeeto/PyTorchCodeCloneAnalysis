import torch.nn.functional as F 
import matplotlib.pyplot as plt
import numpy as np
import torch

data_path = "../ISPRS_VAIHINGEN_SAMPLE/top/top_mosaic_09cm_area1.tif"
label_path = "../ISPRS_VAIHINGEN_SAMPLE/gts/top_mosaic_09cm_area1.tif"

def generate_theta(tr_x = 0.1, tr_y = 0.1, sc_x = 0.0,sc_y = 0.0, rot_x = 1.0 ,rot_y = 1.0):
    affine_matr = np.zeros((2,3),dtype = float)
    affine_matr[0][0] = rot_x
    affine_matr[0][1] = sc_x
    affine_matr[0][2] = tr_x
    affine_matr[1][0] = sc_y
    affine_matr[1][1] = rot_y
    affine_matr[1][2] = tr_y

    return affine_matr

def stn(x,theta):
    ''''
    perform affine transformations using pytorch
    '''
    print("x size",x.size())
    grid = F.affine_grid(theta, x.size())
    x = F.grid_sample(x, grid)
    return x

'''
#apply affine transforma
#0.1
aff_matr = generate_theta(tr_x = -0.15, tr_y = -0.15, sc_x = 0.0,sc_y = 0.0, rot_x = 1.0 ,rot_y = 1.0)
aff_matr = torch.from_numpy(np.expand_dims(aff_matr,axis = 0))
'''

#read the labels
label_area1 = plt.imread(label_path)[:256,:256,:]
label_area2 = plt.imread(label_path)[20:276,20:276,:]


#read the raw image
data_area1 = plt.imread(data_path)[:256,:256,:]
data_area1 = np.swapaxes(data_area1, 0,2).swapaxes(1,2)
data_area1 = torch.from_numpy(np.expand_dims(data_area1*1/255,axis = 0))

data_1_shifted = stn(data_area1,aff_matr)

shift_list_post = [0.1, 0.2, 0.3,0.4,0.5]
shift_list_neg = [-0.1, -0.2,-0.3,-0.4,-0.5]

#compute the shift and add plots
shift_list = []
for shift in shift_list_post:
    aff_matr = generate_theta(tr_x = shift, tr_y = shift, sc_x = 0.0,sc_y = 0.0, rot_x = 1.0 ,rot_y = 1.0)
    aff_matr = torch.from_numpy(np.expand_dims(aff_matr,axis = 0))
    data_shift = stn (data_area1,aff_matr)
    shift_list.append(data_shift)
    #plt.subplot2grid()

#the negative shifts
shift_list_n = []
for shift in shift_list_neg:
    aff_matr = generate_theta(tr_x = shift, tr_y = shift, sc_x = 0.0,sc_y = 0.0, rot_x = 1.0 ,rot_y = 1.0)
    aff_matr = torch.from_numpy(np.expand_dims(aff_matr,axis = 0))
    data_shift = stn (data_area1,aff_matr)
    shift_list_n.append(data_shift)
    #plt.subplot2grid()

figs, axs = plt.subplots(nrows=3,ncols=5,sharex='all',sharey = 'all')
#plt.subplots_adjust(hspace=0.00)
for i in range (len(shift_list)):
    axs[0][i].imshow((shift_list[i][0].numpy()).swapaxes(0,2).swapaxes(0,1))
    axs[1][i].imshow((shift_list_n[i][0].numpy()).swapaxes(0,2).swapaxes(0,1))
    axs[2][i].imshow(label_area1)
