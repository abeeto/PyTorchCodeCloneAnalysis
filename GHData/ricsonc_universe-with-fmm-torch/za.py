import numpy as np
import scipy as sp
from ipdb import set_trace as st
import matplotlib.pyplot as plt
from nbodykit import cosmology as cm
# from lenstools import GaussianNoiseGenerator



def fftind(size):
    k_ind = np.mgrid[:size, :size] - int( (size + 1)/2 )
    k_ind = sp.fft.fftshift(k_ind)
    return( k_ind )


Plin60 = cm.LinearPower(cm.Planck15, redshift=60, transfer = 'CLASS')

def gaussian_random_field(alpha = 3.0,
                          size = 128, 
                          flag_normalize = True, mul = 5000.0):
    """ Returns a numpy array of shifted Fourier coordinates k_x k_y.
        
        Input args:
            alpha (double, default = 3.0): 
                The power of the power-law momentum distribution
            size (integer, default = 128):
                The size of the square output Gaussian Random Fields
            flag_normalize (boolean, default = True):
                Normalizes the Gaussian Field:
                    - to have an average of 0.0
                    - to have a standard deviation of 1.0
        Returns:
            gfield (numpy array of shape (size, size)):
                The random gaussian random field
                
        Example:
        import matplotlib
        import matplotlib.pyplot as plt
        example = gaussian_random_field()
        plt.imshow(example)
        """
        
        # Defines momentum indices
    k_idx = fftind(size)

        # Defines the amplitude as a power law 1/|k|^(alpha/2)
    k__ = k_idx[0]**2 + k_idx[1]**2 + 1e-10

    # amplitude = np.power( k__, -alpha/4.0 )
    amplitude = Plin60(k__/k__.max() * mul) #about right... max should be ~100? #was 5K for 2^11
    # amplitude[0,0] = 0
    
        # Draws a complex gaussian random noise with normal
        # (circular) distribution
    noise = np.random.normal(size = (size, size)) \
        + 1j * np.random.normal(size = (size, size))
    
        # To real space
    gfield = np.fft.ifft2(noise * amplitude).real
    
        # Sets the standard deviation to one
    if flag_normalize:
        gfield = gfield - np.mean(gfield)
        gfield = gfield/np.std(gfield)
        
    return gfield

# for size in [128, 256, 512, 1024, 2048, 4096]:
#     x = gaussian_random_field(size=size)
#     plt.imshow(x)
#     plt.show()

x = gaussian_random_field(size=2**11, alpha=3.0)
#x = gaussian_random_field(size=64)
x = 0.2*gaussian_random_field(size=2**11, mul = 40000) + 0.4*gaussian_random_field(size=2**11, mul = 20000) + 0.6*gaussian_random_field(size=2**11, mul = 10000) + gaussian_random_field(size=2**11, mul = 5000) + gaussian_random_field(size=2**11, mul = 2500) + gaussian_random_field(size=2**11, mul = 1000) 
plt.imshow(x); plt.show()

#np.save('init.npy', x)
# np.save('init_big.npy', x)
# np.save('init_ms.npy', x)
st()
exit()
#in MPc
#from 0.1 Mpc up to 100 Mpc... outputs must also be?? 

N1 = 1000
k1 = np.linspace(0.1, 100.0, N1)
R1 = np.random.randn(N1) + 1j*np.random.randn(N1)

Plin100 = cm.LinearPower(cm.Planck15, redshift=60, transfer = 'CLASS')
Pk1 = Plin100(k1)
psi1 = np.sqrt(Pk1)/np.square(k1) * R1

sq1 = sp.fft.ifft(psi1)
# plt.plot(psi); plt.show() #looks pretty decent

N = 10000
k = np.linspace(0.0, 100.0, N)
R = np.random.randn(N,N) + 1j*np.random.randn(N,N)

xs, ys = np.meshgrid(k, k, indexing='xy')

k = np.sqrt(xs**2 + ys**2)

Pk = Plin100(k)
psi = np.sqrt(Pk)/np.square(k) * R
psi[0,0] = 0

sq = sp.fft.ifft2(psi).real

# plt.imshow(sq); plt.show()

foo = np.diff(np.diff(sq, axis = 0), axis = 1)
plt.imshow(foo); plt.show()

st()

# r1, r2 = np.random.randn(2)


#R1, R2 = sy

Plin60 = cm.LinearPower(cm.Planck15, redshift=60, transfer = 'CLASS')
Plin20 = cm.LinearPower(cm.Planck15, redshift=20, transfer = 'CLASS')


k = np.logspace(-1, 2, 100)

# plt.loglog(k, Plin20(k), c='r')
# plt.loglog(k, Plin60(k), c='g')
Pk = Plin100(k)
psi = np.sqrt(Pk)/np.square(k)

#plt.loglog(k, Plin100(k), c='b')
plt.loglog(k, psi, c='b')

plt.xlabel(r"$k$ $[h \mathrm{Mpc}^{-1}]$")
plt.ylabel(r"$P$ $[h^{-3} \mathrm{Mpc}^{3}]$")
plt.show()



#2 pi / B = 0.1 Mpc
# pi N^1/2 / B = 100 Mpc
