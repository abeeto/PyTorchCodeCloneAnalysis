from aljpy import arrdict, dotdict
import torch
import pybbfmm
from ipdb import set_trace as st
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from munch import Munch as M
import imageio as ii
from time import time
import numpy as np
plt.style.use('dark_background')

class Gravity:
    def __init__(self, plummer = 1E-2, epsilon = 1E-12, **kwargs):
        self.plummer = plummer
        self.gconst = None #need to set....
        self.epsilon = epsilon
        self.chebN = 4 
        self.capacity = 12 
        self.correction = self.compute_correction(**kwargs)

    def compute_correction(self, R = 30, resolution = 51):

        xtarget = ytarget = torch.linspace(0, 0.5, resolution).cuda()
        xyt = torch.stack(torch.meshgrid(xtarget, ytarget), dim = -1).reshape(-1, 2)
        # Tx2 correction points

        xsource = ysource = torch.linspace(-R, R, 2*R+1).cuda()
        xys = torch.stack(torch.meshgrid(xsource, ysource), dim = -1).reshape(-1, 2)
        M = (2*R+1)**2
        # Mx2 source points
        
        rval = torch.zeros(resolution**2)
        for i, xy in enumerate(xyt):

            diff = xys-xy #source minus target...
            sqdists = (diff**2).sum(dim = -1)
            mask = sqdists < (R-0.1)**2
            mask[M//2] = False
            
            xys_ = xys[mask]
            sqdists_ = sqdists[mask]
            diffx = diff[mask][:,0] #only care about x
            forcex = diffx / sqdists_**1.5
            rval[i] = forcex.sum()

        #index by yx
        rval = rval.reshape(1,1,resolution,resolution).permute(0,1,3,2)
        # plt.imshow(rval[0,0].numpy()); plt.show() 
        return rval.cuda()

    def brute_force(self, xy, R = 30): #this implementation is correct
        #xy is ?x2

        xoff = yoff = torch.linspace(-R, R, 2*R+1).cuda()
        xyoff = torch.stack(torch.meshgrid(xoff, yoff), dim = -1).reshape(-1, 2)
        #Nx2

        total = 0

        for off in xyoff:
            xy_ = xy + off
            sqdist = (xy_**2).sum(-1)

            total += torch.where(
                torch.max(sqdist < self.epsilon, R-0.9 < sqdist), #if either of these...
                torch.zeros_like(sqdist),
                -xy_[...,0]/( torch.sqrt(sqdist)*sqdist ),
            )

        return total
    
    def direct(self, xy): 
        sqdist = (xy**2).sum(-1)
        return torch.where(
            sqdist < self.epsilon,
            torch.zeros_like(sqdist), #0...
            -xy[...,0]/( torch.sqrt(sqdist)*(sqdist + self.plummer**2) )
        )

    def indirect(self, xy):
        #requires xy in [0,0.5]^2
        if len(xy.shape) == 2: #needs to support rank 2
            return self.indirect(xy.unsqueeze(0))[0]

        return F.grid_sample(
            self.correction, 
            (xy*4-1).unsqueeze(0), #needs to be rescaled to -1,1, NHW2
            mode='bilinear',
            padding_mode='border', #shouldn't matter
            align_corners = False,
        )[0,0]
    
    def gravityx(self, a, b):
        target = a
        source = b
        
        xy = (target-source) % 1 #get everything to top right quadrant...
        x = xy[...,0]
        y = xy[...,1]

        # try:
        #     foo = (xy**2).sum(-1)[(xy**2).sum(-1) > 0].min()
        #     print(foo)
        #     if foo.item() < 0.02:
        #         st()
        # except:
        #     pass

        '''
        taking advantage of symmetry: 
        x must be in [-1,1]^2, in one of the four quadrants
        we can always flip x over the horizontal axis for free (since we're computing x-force)
        we can flip x over the vertical axis, and just flip the sign of the force
        by this, we can always move x to be in [0,0.5]^2 of the top right quadrant
        this fact will be used below
        '''

        horflip = y > 0.5 #if y > 0.5, flip over horizontal
        verflip = x > 0.5 #if x > 0.5, flip over vertical axis

        y = torch.where(horflip, (1-y), y)
        x = torch.where(verflip, (1-x), x)
        sign = torch.where(verflip, -torch.ones_like(x), torch.ones_like(x))
        xy = torch.stack((x, y), dim=-1)
        return sign * (self.direct(xy) + self.indirect(xy))
        # return sign * self.direct(xy)
        # return sign * self.brute_force(xy) #for debugging...

    def gravityy(self, a, b):
        #idea -- flip x and y coordinates, then the rest is the same
        return self.gravityx(a.flip(-1), b.flip(-1)) 

    def compute(self, points):
        if points.shape[0] <= 4096:
            return self.compute_brute(points)
        
        prob = arrdict.arrdict(
            sources=points,
            charges=torch.ones(points.shape[0]).cuda(),
            targets=points,
        )
        #Nx2 output
        return self.gconst * self.solve_multikernel(prob, [self.gravityx, self.gravityy])
        # rval = self.gconst * self.solve_multikernel(prob, [self.gravityx, self.gravityy])
        # gt = self.compute_brute(points)
        # rval /= self.gconst
        # gt /= self.gconst
        
        # diff = rval-gt
        # mags = torch.sqrt((gt**2).sum(-1))
        # diffmags = torch.sqrt((diff**2).sum(-1))

        # reldiff = diffmags/mags
        # print(reldiff.max())
        # st()
        # return rval * self.gconst

    def compute_brute(self, points): 
        return self.gconst * torch.stack(
            (self.gravityx(points.unsqueeze(1), points.unsqueeze(0)),
             self.gravityy(points.unsqueeze(1), points.unsqueeze(0))),
            dim = -1
        ).sum(dim=1) #sum out the sources
    
    def solve_multikernel(self, prob, kernels):
        from pybbfmm import chebyshev, orthantree, scale
        
        cheb = chebyshev.Chebyshev(self.chebN, prob.sources.shape[1], device=prob.sources.device)
        prob.kernel = lambda a, b: None
        scaled = scale(prob)
        
        tree, indices, depths = orthantree.orthantree(scaled, capacity=self.capacity)
        scheme = orthantree.interaction_scheme(tree, depths, PERIODIC=True)

        outputs = []
        for kernel in kernels:

            scaled_ = M(**scaled) #copy scaled...
            scaled_.kernel = lambda a, b: kernel(a*scaled.scale, b*scaled.scale)
            
            outputs.append(pybbfmm.evaluate(**dotdict(
                cheb=cheb, 
                scaled=scaled_,
                tree=tree, 
                scheme=scheme,
                indices=indices,
                depths=depths
            )))

        return torch.stack(outputs, axis = -1)

class Universe:
    def __init__(self, uargs, debug = False):

        self.debug = debug
        self.uargs = uargs
        self.gcomputer = Gravity()
        self.history = []

    def initialize(self, version=None):

        N = self.uargs.nbodies
        
        if version == 0:
            self.pos = torch.rand(N, 2).cuda() * 0.3 + 0.4
            self.vel = (torch.rand(N, 2).cuda()-0.5) * 0.0

        elif version == 1:

            M = int(N**0.5)
            z = torch.linspace(0, (M-1)/M, M).cuda()
            self.pos = torch.stack(torch.meshgrid(z, z), dim = -1).reshape(-1, 2) + 0.0
            self.vel = torch.zeros_like(self.pos)

        else:

            M = int(N**0.5)
            z = torch.linspace(0, (M-1)/M, M).cuda()
            
            self.pos = torch.stack(torch.meshgrid(z, z), dim = -1).reshape(-1, 2) + (0.5/M)
            # self.pos += (torch.rand(N, 2).cuda()-0.5) * 1E-4 #grid spacing is 1E-4 so...

            #density = np.load('init.npy')
            #density = np.load('init_big.npy')
            density = np.load('init_ms.npy')
            grad = torch.Tensor(np.stack(np.gradient(density), axis = -1)).cuda().reshape(-1, 2)
            self.pos += grad * 1E-4 #was 1E-3 before..., 1E-2 also works well
            self.pos %= 1
            
            #"glass"
            #CHANGE to 0.5e-4
            
            self.vel = torch.zeros_like(self.pos)

        self.last_a = torch.zeros_like(self.pos)
        self.time = 0

        if self.debug:
            self.history.append(self.pos)

        self.initializing = False
        self.initializing_steps = 0

    def scale(self, t = None):
        if t is None:
            t = self.time
        time = self.uargs.start_time + t * self.uargs.timestep
        scale = 0.693 * np.sinh(time / 3.42E+17)**(2/3)
        return scale

    def step(self):

        #1. compute gravitational constant
        # current_size = self.scale() * self.uargs.size
        current_size = self.uargs.size
        ud_over_m = current_size
        ut_over_s = self.uargs.timestep
        um_over_kg = (self.uargs.density*self.uargs.size**2) / self.uargs.nbodies
        gravity_u = self.uargs.gravity * (um_over_kg * ut_over_s**2) / ud_over_m**3
        self.gcomputer.gconst = gravity_u

        scale = self.scale()
        self.gcomputer.plummer = self.uargs.plummer / (ud_over_m*scale) #<- scale on the bottom..
        print(self.gcomputer.plummer)

        self.pos += 0.5 * self.vel / scale**2
        self.pos %= 1

        rand_offset = torch.rand(1,2).cuda()
        #should not be affected by random displacement of all points
        a = self.gcomputer.compute((self.pos + rand_offset)%1) 
        self.vel += a / scale
        self.pos += 0.5 * self.vel / scale**2
        
        #2. update universe via leapfrog -- with dynamic stepping?
        #KDK-form
        # self.vel += self.last_a/2 #v(+0.5)
        # self.pos += self.vel #x(+1)
        # self.pos %= 1

        # a = self.gcomputer.compute(self.pos)
        
        # if self.initializing:
        #     a = -a #reverse forces while initializing
            
        # self.vel += a/2 #v(+1)
        # self.last_a = a

        speeds = torch.sqrt((self.vel**2).sum(-1))/scale**2

        # forces = torch.sqrt((a**2).sum(-1))
        # print('forces --', forces.mean(), forces.max())

        # if forces.mean().item() < 1E-7:
        #     print('done initializing, starting to simulate...')
        #     self.initializing = False

        # print(self.scale() / self.scale(self.time+1))
        #speeds = torch.sqrt((torch.min(self.vel.abs(), 1 - self.vel.abs())**2).sum(-1))
        print('speeds -- ', speeds.mean(), speeds.max())

        if not self.initializing:
            # self.vel *= self.scale() / self.scale(self.time+1) #scale velocity down to account for increased scale
            self.time += 1
            #only increment time and decrease velocity if not initializing...
        else:
            self.vel *= 0.0
            self.initializing_steps += 1

        if self.debug:
            self.history.append(self.pos)
        
        if not self.debug and (self.time % 10 == 0):
            np.save(f'out/{self.time:04d}', self.pos.cpu().numpy())

    def plot_history(self):
        hist = torch.stack(U.history, axis = 0).cpu().numpy()

        for j in range(U.uargs.nbodies):
            foo = plt.scatter(hist[500::1,j,0], hist[500::1,j,1], alpha = 0.2, s=1)
            # plt.scatter([hist[-1,j,0]], [hist[-1,j,1]], c = foo[0].get_color())

        plt.axes().set_aspect('equal')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def plot(self):
        pos = self.pos.cpu().numpy() % 1
        xs = pos[...,0]
        ys = pos[...,1]
        plt.scatter(xs, ys, s=1)
        plt.axes().set_aspect('equal')
        plt.tight_layout()
        plt.axis('off')
        plt.show()

    def go(self):
        self.initialize()

        # self.plot()
        for i in range(10000):

            # if i % 10 == 0:
            #     self.plot()
                
            t0 = time()
            self.step()
            print(f'step {i} in {time()-t0:.2f} seconds')

        # self.plot_history()

#tweak plummer as necessary...
        
uargs = M(
    density = 0.02, #kg/m^2 <- obtained by projecting down 1E-26 kg/m^3 into one less dimension
    size = 2E+24, #length of our cube, in meters
    plummer = 2E+21, #plummer smoothing radius, same as in mill.run #5 good, 1 bad
    gravity = 6.67E-11, #gravitational constant
    hubble = 2.25E-18, #in hertz...
    simtime = 5E+17, #seconds
    timestep = 5E+13, #seconds
    start_time = 5E+15, #seconds #the lower this is, faster webbing and bad shapes
    nbodies = 2**22, #number of objects
)

#adaptive stepping needed..

torch.manual_seed(0)
U = Universe(uargs, debug = False)
U.go()
# U.plot_history()

st()

#~/venvs/torchnew/lib/python3.8/site-packages/pybbfmm/
