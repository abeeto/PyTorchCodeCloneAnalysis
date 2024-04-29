import numpy as np
from math import pi
#from quats import Quat
import torch
from quats import outer_prod, rand_quats, rotate

#rotate by 0 or 180 degrees about x axis
hcp_r1 = torch.eye(4)[:2]


# rotate about 0, 60, ... 300 degrees about z axis
hcp_r2 = torch.zeros((6,4))
hcp_r2[:,0] = torch.cos(torch.arange(6)/6*pi)
hcp_r2[:,3] = torch.sin(torch.arange(6)/6*pi)
hcp_syms = outer_prod(hcp_r1,hcp_r2).reshape((-1,4))

#hcp_syms = hcp_r2.outer_prod(hcp_r1).transpose((1,0)).reshape(-1)


# rotate about diagonal on cube
fcc_r1 = torch.zeros((3,4))
fcc_r1[:,0] = torch.cos(2/3*pi*torch.arange(3))
fcc_r1[:,1:] = (torch.sin(2/3*pi*torch.arange(3))/(3**0.5))[:,None]

# rotate by 0 or 180 degrees around x-axis
#fcc_r2 = torch.array(np.eye(4)[:2])
fcc_r2 = torch.eye(4)[:2]

fcc_r3 = torch.zeros((4,4))
fcc_r3[:,0] = torch.cos(pi/4*torch.arange(4))
fcc_r3[:,3] = torch.sin(pi/4*torch.arange(4))

fcc_r12 = outer_prod(fcc_r1,fcc_r2)
fcc_syms = outer_prod(fcc_r12,fcc_r3).reshape((-1,4))



if __name__ == '__main__':

    from plotting_utils import *

    np.random.seed(1)
    q1 = rand_quats(())

    rhomb_wire = path2prism(rhomb_path)
    all_rots = outer_prod(q1,hcp_syms)
    all_wires = rotate(all_rots,rhomb_wire)
    all_axes = rotate(all_rots,rhomb_axes)
    
    def setup_axes(m,n):
        r = np.sqrt(2)
        fig = plt.figure()
        axes = [fig.add_subplot(m,n,i+1,projection='3d') for i in range(m*n)]
        for a in axes:
            a.set_xlim(-r,r)
            a.set_ylim(-r,r)
            a.set_zlim(-r,r)    
        return fig, axes

    fig, axes = setup_axes(3,4)

    for i, ax in enumerate(axes):

        ax.plot(*all_wires[0].T,color='#888')
        ax.plot(*all_wires[i].T,color='#000')
        plot_axes(ax,all_axes[i])


    square_wire = path2prism(square_path)

    all_rots = outer_prod(q1,fcc_syms)
    all_wires = rotate(all_rots,square_wire)
    all_axes = rotate(all_rots,square_axes)


    fig, axes = setup_axes(4,6)

    for i, ax in enumerate(axes):
        ax.plot(*all_wires[0].T,color='#888')
        ax.plot(*all_wires[i].T,color='#000')
        plot_axes(ax,all_axes[i])


    plt.show()



