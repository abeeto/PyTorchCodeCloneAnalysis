import torch
import numpy as np
import time




class Propagator:
    @staticmethod
    def _get_grid(nx, ny, dx, dy):
        # generate spacial grid
        x = (np.arange(nx) - (nx - 1.) / 2) * dx
        y = (np.arange(ny) - (ny - 1.) / 2) * dy
        yy, xx = np.meshgrid(y, x)
        return xx, yy

    @staticmethod
    def _get_frequencies(nx, ny, dx, dy):
        k_x = np.fft.fftfreq(n=nx, d=dx) * 2 * np.pi
        k_y = np.fft.fftfreq(n=ny, d=dy) * 2 * np.pi
        k_Y, k_X = np.meshgrid(k_y, k_x, indexing='xy')
        return k_X, k_Y

    @staticmethod
    def _setup_H_tensor(nx, ny, k, z_list, dx, dy, limit_bw):
        # define spatial frequency vectors
        k_x = np.fft.fftfreq(n=nx, d=dx) * 2 * np.pi
        k_y = np.fft.fftfreq(n=ny, d=dy) * 2 * np.pi
        k_Y, k_X = np.meshgrid(k_y, k_x, indexing='xy')
        k_Y = k_Y[None, :, :]
        k_X = k_X[None, :, :]

        z_list = np.array(z_list)
        z_list = z_list[:, None, None]

        # define wavenumber for each wavevector in the direction of propagation
        k_z = np.sqrt(0j + k ** 2 - k_X ** 2 - k_Y ** 2)
        k_z = torch.tensor(k_z)

        phase = k_z * z_list
        H = torch.exp(1j * phase)

        if limit_bw:
            # Apply antialias filter to H
            # See paper 'Band-Limited Angular Spectrum Method for Numerical SImulation of Free-Space Propagation in Far and near fields'
            del_f_x = 1. / (2. * nx * dx)
            del_f_y = 1. / (2. * ny * dy)

            k_x_limit = k / np.sqrt((2 * del_f_x * z_list) ** 2 + 1)
            k_y_limit = k / np.sqrt((2 * del_f_y * z_list) ** 2 + 1)

            kx_mask = (k_X / k_x_limit) ** 2 + (k_Y / k) ** 2 <= 1.
            ky_mask = (k_Y / k_y_limit) ** 2 + (k_X / k) ** 2 <= 1.

            comb_mask = np.logical_and(kx_mask, ky_mask)
            comb_mask = torch.tensor(comb_mask)
            comb_mask = comb_mask.int().float()  # convert boolean mask to floats

            H *= comb_mask

        return H

    def __init__(self, nx, ny, k, z_list, dx, dy, limit_bw=True, **_):
        self.H = Propagator._setup_H_tensor(nx, ny, k, z_list, dx, dy, limit_bw)
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy


        # grid isn't required for angular propagation but is useful for other operations
        self.xx, self.yy = Propagator._get_grid(ny, ny, dx, dy)

    def prop(self, field):
        assert field.size()[0] == self.nx and field.size()[1] == self.ny


        # fourier transform the input field
        U = torch.fft.fft2(field)
        U = U[None, :, :]

        E_k_prop = U * self.H
        E_prop = torch.fft.ifft2(E_k_prop)

        return E_prop


class PropagatePadded(Propagator):
    def __init__(self, nx, ny, k, z_list, dx, dy, limit_bw=True, pad_factor=1., **_):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.xx, self.yy = Propagator._get_grid(nx, ny, dx, dy)

        self.pad_factor = pad_factor
        nx_padded = nx + 2 * int(nx * pad_factor / 2)
        ny_padded = ny + 2 * int(ny * pad_factor / 2)
        self.propagator = Propagator(nx_padded, ny_padded, k, z_list, dx, dy, limit_bw)

    @staticmethod
    def _pad(source, pad_factor=1.):
        n_x, n_y = source.size()
        pad_x = int(n_x * pad_factor / 2)
        pad_y = int(n_y * pad_factor / 2)
        return torch.nn.functional._pad(input=source, pad=(pad_x, pad_x, pad_y, pad_y), mode='constant', value=0)

    @staticmethod
    def _unpad(source, pad_factor=1.):
        if pad_factor == 0.:
            return source
        else:
            *_, n_x, n_y = source.size()
            pad_x = int(n_x * pad_factor / (2 + 2 * pad_factor))
            pad_y = int(n_y * pad_factor / (2 + 2 * pad_factor))
            return source[:, pad_x:-pad_x, pad_y:-pad_y]

    def prop(self, field):
        assert field.size()[0] == self.nx and field.size()[1] == self.ny
        field = PropagatePadded._pad(field, pad_factor=self.pad_factor)
        field = self.propagator.prop(field)
        field = PropagatePadded._unpad(field, pad_factor=self.pad_factor)

        return field





if __name__ == "__main__":
    shape = (2000, 2000)
    dx = 350e-9
    dy = 350e-9
    f = 100e-6
    k = 2 * np.pi / 633e-9

    p = Propagator(*shape, k, z_list=[f, ], dx=dx, dy=dy, limit_bw=True)



    field = np.exp(-1j * k * np.sqrt(p.xx**2 + p.yy**2 + f**2))
    field = torch.tensor(field, requires_grad=True)

    output = p.prop(field)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(np.abs(output.detach()[0, :, :])**2)
    plt.show()
