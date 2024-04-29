import numpy as np
from ipdb import set_trace as st
import numdifftools as nd
import matplotlib.pyplot as plt

def compute_exact(xy, N=30):
    #this is R^2 -> R
    #we want to find a taylor expansion of the function
    #N is a radius..

    xs = np.linspace(-N, N, 2*N+1)
    ys = np.linspace(-N, N, 2*N+1)

    xmg, ymg = np.meshgrid(xs, ys, indexing='ij') #fix me...

    ymg = ymg.reshape(-1,)
    xmg = xmg.reshape(-1,)

    mg = np.stack((xmg, ymg), -1) #Nx2

    #filter out points as follows...
    # dist = mg -
    M = (2*N+1)**2
    mask = np.ones(M, dtype = np.bool)
    mask[M//2] = False
    
    sqdists = ((mg - xy)**2).sum(axis = -1)
    mask[sqdists >= (N-0.9)**2] = False

    mgv = mg[mask]
    #valid mask...
    direction = mgv-xy
    Z = (direction**2).sum(-1)[:,np.newaxis]**(1.5)
    force = direction / Z
    force = force.sum(0)[0] #x-force
    return force

# x = compute_exact(np.array([0.5,0.0]))
# print(x)

# grad = approx_fprime(np.array([0.0, 0.0]), compute_exact, 1E-6)
# print(grad)

gradient = nd.Gradient(compute_exact)#, step=1E-6)
hessian = nd.Hessian(compute_exact)#, step=1E-6)

# center = np.array([0.5, 0.25])
# center = np.array([0.25, 0.5])

center = np.array([0.4, 0.4])
# center = np.array([0.1, 0.1]) #<skip?
# center = np.array([0.1, 0.4])
# center = np.array([0.4, 0.1])

#so... the 5 regions?
#if close enough...

offset = compute_exact(center)
grad = gradient(center)
hess = hessian(center)

def approxilator(xy):
    diff = xy-center
    return offset + (grad.T @ diff) + (diff.T @ hess @ diff)/2

def centerforce(xy):
    return xy[0]/(np.linalg.norm(xy) + 1E-5)**3

def error(xy):
    exact = compute_exact(xy)
    approx = approxilator(xy)

    total = centerforce(xy)

    relative_error = abs(exact-approx)/total
    return relative_error

def plot_error(): #what should this look like, actually?
    Q = 101
    out = np.zeros((Q,Q))
    for i, x in enumerate(np.linspace(0.0, 0.5, Q)):
        for j, y in enumerate(np.linspace(0.0, 0.5, Q)):
            out[j,i] = error(np.array([x, y])+1E-5)
            # out[j,i]=compute_exact(np.array([x,y])+1E-5) - centerforce(np.array([x,y])+1E-5)
            # out[j,i]=compute_exact(np.array([x,y])+1E-5)

    plt.imshow(np.clip(out, 0.0, 1.0))
    # plt.imshow(out)
    plt.show()

# zero = np.array([0,0])
# print(approxilator(zero))

plot_error()
st()
