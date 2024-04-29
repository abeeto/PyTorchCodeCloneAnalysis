import numpy as np
import matplotlib.pyplot as plt
from magtorch import *

#skmodes = np.load("modes2_skL_48_Lz_64.npz")
#spmodes = np.load("modes2_spL_48_Lz_64.npz")
k = 1.233
h = 0.05
D = 0.268
skmodes = np.load("modes_k_1.233_h_0.050_skL_48_Lz_96.npz")
spmodes = np.load("modes_k_1.233_h_0.050_spL_48_Lz_96.npz")
modes = skmodes['arr_1'].reshape(1,2,48,48,96,-1)
modes_sp= spmodes['arr_1'].reshape(1,2,48,48,96,-1)
modeden = np.sqrt(modes[0,0]**2 + modes[0,1]**2)
config = np.load("testk_1.233_h_0.050_L_48_Lz_96_config.npy")
energy = np.load("testk_1.233_h_0.050_L_48_Lz_96_energypath.npy")
#ts = np.load("../testk_1.233_h_0.050_L_48_Lz_96_ts.npy")
#energy = energy - energy.max()
#plt.plot(ts,energy/np.abs(energy).max(), '.')
#plt.plot()
plt.show()
#config = np.load("../test2L_48_Lz_64_config.npy")
#energy = np.load("../test2L_48_Lz_64_energypath.npy")
#modeden = modes[0,1]
#plt.imshow(modeden[:,:,10,7])
x =np.linspace(-47/2,47/2,48)
y =np.linspace(-47/2,47/2,48)
x,y = np.meshgrid(x,y,indexing = 'ij')
ms = [] 
for j in range(0,130):

    mx, my, mz,modex, modey, modez = modetransform(modes[:,:,:,:,:,j],config[0:1])
    if j == 0:
        showconfig3D(config)
        #showconfig(config, ti = 0, zi = 32)
    #sx = my*modez - mz*modey
    #sy = mz*modex - mx*modez
    #nx,ny = x/np.sqrt(x*x+y*y), y/np.sqrt(x*x+y*y)
    chiral = modez #sx*ny -sy*nx
    print(np.abs(np.array([(np.cos(m*np.arctan2(y,x))*modez).sum() for m in range(0,10)])))
    print(np.abs(np.array([(np.sin(m*np.arctan2(y,x))*modez).sum() for m in range(1,10)])))
    cosmaxind = np.argmax(np.abs(np.array([(np.cos(m*np.arctan2(y,x))*modez).sum() for m in range(0,10)])))
    sinmaxind = np.argmax(np.abs(np.array([(np.sin(m*np.arctan2(y,x))*modez).sum() for m in range(0,10)])))
    if np.abs((np.cos(cosmaxind*np.arctan2(y,x))*modez).sum())>np.abs((np.sin(sinmaxind*np.arctan2(y,x))*modez).sum()):
        ms.append(cosmaxind)
        print(cosmaxind)
    else:
        ms.append(sinmaxind+20)
        print(sinmaxind)
ms = np.array(ms)
for m in range(0,10):
    plt.plot(skmodes['arr_0'][:130][ms == m],'-s',label = "m="+str(m))
plt.xlabel(r"$_z$")
plt.ylabel(r"$E$")
plt.legend()
plt.show()
indsp = np.argmax(energy)
ms = [] 
for j in range(1,130):
    mx, my, mz,modex, modey, modez = modetransform(modes_sp[:,:,:,:,:,j],config[indsp:indsp+1],zi =-1)
    if j == 1:
        showconfig3D(config, ti = indsp)
    #sx = my*modez - mz*modey
    #sy = mz*modex - mx*modez
    #nx,ny = x/np.sqrt(x*x+y*y), y/np.sqrt(x*x+y*y)
    chiral = modez #sx*ny -sy*nx
    print(np.abs(np.array([(np.cos(m*np.arctan2(y,x))*modez).sum() for m in range(0,10)])))
    print(np.abs(np.array([(np.sin(m*np.arctan2(y,x))*modez).sum() for m in range(1,10)])))
    cosmaxind = np.argmax(np.abs(np.array([(np.cos(m*np.arctan2(y,x))*modez).sum() for m in range(0,10)])))
    sinmaxind = np.argmax(np.abs(np.array([(np.sin(m*np.arctan2(y,x))*modez).sum() for m in range(0,10)])))
    if np.abs((np.cos(cosmaxind*np.arctan2(y,x))*modez).sum())>np.abs((np.sin(sinmaxind*np.arctan2(y,x))*modez).sum()):
        ms.append(cosmaxind)
        print(cosmaxind)
    else:
        ms.append(100+sinmaxind)
        print(sinmaxind)
    print(spmodes['arr_0'][j])
    if spmodes['arr_0'][j] < 0.:
        showmode(spmodes['arr_1'][:,j], config[indsp:indsp+1])
        showmode(spmodes['arr_1'][:,j], config[indsp:indsp+1],zi=1)
        showmode(spmodes['arr_1'][:,j], config[indsp:indsp+1],zi=2)
        showmode(spmodes['arr_1'][:,j], config[indsp:indsp+1],zi=3)
        showmode(spmodes['arr_1'][:,j], config[indsp:indsp+1],zi=4)
        showmode(spmodes['arr_1'][:,j], config[indsp:indsp+1],zi=6)
        showmode(spmodes['arr_1'][:,j], config[indsp:indsp+1],zi=8)
        showmode(spmodes['arr_1'][:,j], config[indsp:indsp+1],zi=20)
        #showmode(spmodes['arr_1'][:,j], config[indsp:indsp+1],zi=40)
ms = np.array(ms)
for m in range(0,10):
    plt.plot(spmodes['arr_0'][1:130][ms == m],'-o', label = "m="+str(m))

plt.xlabel(r"$_z$")
plt.ylabel(r"$E$")
plt.legend()
plt.show()
print(skmodes)
plt.plot(skmodes['arr_0'])
plt.plot(spmodes['arr_0'])
plt.show()
skmodes_real = skmodes['arr_0'][2:-1]
spmodes_real = spmodes['arr_0'][3:]
print(np.log(skmodes_real[skmodes_real < (2*h + k)*D**2]).sum() - np.log(spmodes_real[skmodes_real < (2*h + k)*D**2]).sum())
print(np.log(skmodes_real[skmodes_real < (2*h + k)*D**2]).sum() - np.log(spmodes_real[spmodes_real < (2*h + k)*D**2]).sum()-((skmodes_real<(2*h + k)*D**2).sum()-(spmodes_real<(2*h + k)*D**2).sum())*np.log((2*h + k)*D**2))