
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
Lz = 1
Lt = 192#32#
D =  0.268#0.06554346#0.0327#0.13165#0.5#0
lr = 0.3#0.2 # 0.01 gives the best performance
f = open("phase_pointsp")
for line in f.readlines():
    try:
        kh = line.split(" ")
        L = int(kh[0])
        k = float(kh[1])
        h = float(kh[2])
    except:
        continue
    parameters = {'J':1.,'a':1.0,'Dinter': D, 'Dbulk':0., 'Anisotropy': k*D**2, 'H':h*D**2}
    a = Model(L,L,Lz,Lt,parameters=parameters)
    a.config = torch.tensor(skyrmion_timeline(L,L,1/D, 1.9,0.6,Lt),\
        requires_grad = True, device = device,dtype = torch.float)#,16#_bobber
    a.external_field_update()

    ### ----- first run: finding path and saddle point -----

    ts_out = np.linspace(0.,1.,Lt)

    opt_Adam = torch.optim.Adagrad([a.config], lr=lr,lr_decay = 0.5e-6)#0.5e-6
    def closure():
        opt_Adam.zero_grad()
        loss = a.energy_all()
        a.config.grad = torch.autograd.grad(loss, [a.config])[0]
        return loss
    epath = a.energy().to("cpu").detach().numpy()

    for i in tqdm(range(2000)):#5000
        for _ in range(5):
            closure()
            opt_Adam.step()#closure
        a.configvec = flat_map(a.config)
        ts_in = reaction_parameter(a.configvec)
        a.config.data = spherical_map(splinelinear_interpolation3d(a.configvec.data,ts_in,ts_out))
        if i%100 ==0:
            epath = a.energy().to("cpu").detach().numpy()
            print(epath[0])
            print(epath.max())

    epath = a.energy().to("cpu").detach().numpy()
    index_sp = np.argmax(epath)
    target = ts_out[index_sp]
    targetinterval = 0.1
    targetintervalLt = 40
    if (target + targetinterval/2 >1.0):
        print("make interval smaller")
        quit()

    ### ----- second run: increase precision near saddle point -----
    ts_target = np.linspace(target - targetinterval/2,target + targetinterval/2,targetintervalLt)
    ts_out = np.linspace(0.,1-targetinterval,Lt-targetintervalLt)
    ts_out[ts_out>target - targetinterval/2] += targetinterval
    ts_out = np.sort(np.concatenate([ts_out,ts_target]))
    opt_Adam = torch.optim.Adagrad([a.config], lr=lr)
    def closure():
        opt_Adam.zero_grad()
        loss = a.energy_all()
        a.config.grad = torch.autograd.grad(loss, [a.config])[0]
        return loss

    for i in tqdm(range(5000)):#50
        for _ in range(5):
            closure()
            opt_Adam.step()#closure
        a.configvec = flat_map(a.config)
        ts_in = reaction_parameter(a.configvec)
        a.config.data = spherical_map(splinelinear_interpolation3d(a.configvec.data,ts_in,ts_out))
        if i%100 ==0:
            epath = a.energy().to("cpu").detach().numpy()
            print(epath[0])
            print(epath.max())

    index_sp = np.argmax(a.energy().to("cpu").detach().numpy())
    a.configvec = flat_map(a.config)
    ts_out_d = reaction_parameter(a.configvec)

    a.save_model("../testXk_%4.3f_h_%4.3f_L_%d_Lz_%d" %(k,h,L,Lz))
    np.save("../testXk_%4.3f_h_%4.3f_L_%d_Lz_%d_ts" %(k,h,L,Lz),ts_out)

    ### ----- solveing eigenmodes at saddle point and minimun -----

    epath = a.energy().to("cpu").detach().numpy()
    config_sk = a.config[0:1].data.clone()
    config_sp = a.config[index_sp:index_sp+1].data.clone()
    del a

    t1 = time.time()
    e1, v1 = sparse_modes(config_sk,parameters,nmodes=250)
    print("skyrmion time (truncate):", time.time()-t1)
    print(e1)
    np.savez("modes_Xk_%4.3f_h_%4.3f_skL_%d_Lz_%d"%(k,h,L,Lz),e1,v1)


    t1 = time.time()
    e1, v1 = sparse_modes(config_sp,parameters,sigma=-10.0,nmodes=250)
    print("saddle-point time (truncate):", time.time()-t1)
    print(e1)
    np.savez("modes_Xk_%4.3f_h_%4.3f_spL_%d_Lz_%d"%(k,h,L,Lz),e1,v1)
