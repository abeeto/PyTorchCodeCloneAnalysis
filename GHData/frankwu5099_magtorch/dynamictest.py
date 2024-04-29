
from magtorch import *
import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import time
from tqdm import tqdm

## what should be saved
## 1. config (numpy array)
## 2. parameters? yaml 
## 3. enrgy pash (numpy array)
## 4. k modes (energy & mode) for skyrmion and saggle point (2 files npz (energy and modes))
#L = 32
Lz = 24
Lt = 1
D = 0.268
lr = 0.3#0.2 # 0.01 gives the best performance
for ri in range(1):
    L = 64
    k = 1.3
    h = 0.1
    parameters = {'J':1.,'a':1.0,'Dinter': D, 'Dbulk':0., 'Anisotropy': k*D**2, 'H':h*D**2}
    a = Model(L,L,Lz,Lt,parameters=parameters)
    a.config = torch.tensor(skyrmion_bobber(L,L,Lz,7, 2.0,0.6,0),\
        requires_grad = True, device = device,dtype = torch.float)#,16#_bobber
    a.config = torch.unsqueeze(a.config, 0)
    a.external_field_update()
    #read config
    #a.config = torch.tensor(np.load("../testk_1.233_h_0.050_L_48_Lz_96_config.npy"),requires_grad = True, device = device,dtype = torch.float)

    ts_out = np.linspace(0.,1.,Lt)
    print(a.energy().to("cpu").detach().numpy())
    for i in range(2000):
        LLG_rk4(a, 0.05, 0.3, 1)
        torch.cuda.empty_cache()
    print(a.energy().to("cpu").detach().numpy())
    showconfig3D(a.config.to("cpu").detach().numpy())
    for i in range(2000):
        LLG_rk4(a, 0.05, 0.3, 1)
        torch.cuda.empty_cache()
    print(a.energy().to("cpu").detach().numpy())
    showconfig3D(a.config.to("cpu").detach().numpy())
    for i in range(4000):
        LLG_rk4(a, 0.05, 0.3, 1)
        torch.cuda.empty_cache()
    print(a.energy().to("cpu").detach().numpy())
    showconfig3D(a.config.to("cpu").detach().numpy())
    for i in range(4000):
        LLG_rk4(a, 0.05, 0.3, 1)
        torch.cuda.empty_cache()
    print(a.energy().to("cpu").detach().numpy())
    showconfig3D(a.config.to("cpu").detach().numpy())
    for i in range(4000):
        LLG_rk4(a, 0.05, 0.3, 1)
        torch.cuda.empty_cache()
    print(a.energy().to("cpu").detach().numpy())
    showconfig3D(a.config.to("cpu").detach().numpy())
        
    #for zi in range(0,1,8):
    #    showconfig(a.config.detach().to("cpu").numpy(),ti =20,zi =zi)
