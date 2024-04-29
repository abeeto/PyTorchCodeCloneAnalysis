import torch
import numpy as np
import torch.nn.functional as F

from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm
from torch import nn


class Generator(nn.Module):
    def __init__(self, n_features, z_dim, angles):
        super(Generator, self).__init__()
        self.tanh = nn.Tanh()
        self.out_ch = 3
        self.ups_3d = nn.Upsample(scale_factor=2, mode='nearest')
        self.ups_2d = nn.Upsample(scale_factor=2, mode='nearest')

        xstart = (torch.randn((1, n_features, 4, 4, 4)) - 0.5) / 0.5
        nn.init.xavier_uniform(xstart.data, 1.)
        self.xstart = nn.Parameter(xstart)
        self.xstart.requires_grad = True

        self.rb1 = GenResBlockNdim(n_features, n_features // 2, n_dims=3)
        self.adain_1, self.z_mlp1 = self._adain_module_3d(z_dim, n_features // 2)
        self.rb2 = GenResBlockNdim(n_features // 2, n_features // 4, n_dims=3)
        self.adain_2, self.z_mlp2 = self._adain_module_3d(z_dim, n_features // 4)

        self.postproc = nn.Sequential(
            nn.Conv3d(n_features // 4, n_features // 8, kernel_size=3, padding=1),
            nn.InstanceNorm3d(n_features // 8, affine=True),
            nn.ReLU(),
            nn.Conv3d(n_features // 8, n_features // 8, kernel_size=3, padding=1),
            nn.InstanceNorm3d(n_features // 8, affine=True),
            nn.ReLU()
        )

        pnf = (n_features // 8) * (4 ** 2) # 512
        self.proj = nn.Sequential(
            nn.Conv2d(pnf, pnf//2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(pnf//2, affine=True),
            nn.ReLU()
        )  # should be 1x1
        self.rb1_2d = GenResBlockNdim(pnf // 2, pnf // 4, n_dims=2)
        self.adain_3, self.z_mlp3 = self._adain_module_2d(z_dim, pnf // 4)
        self.rb2_2d = GenResBlockNdim(pnf // 4, pnf // 8, n_dims=2)
        self.adain_4, self.z_mlp4 = self._adain_module_2d(z_dim, pnf // 8)
        self.conv_final = nn.Conv2d(pnf // 8, self.out_ch, 3, padding=1)

        # calc theta angles
        self.angles = self._angles_to_dict(angles)
        self.rot2idx = {
            'x': 0,
            'y': 1,
            'z': 2
        }

    def _to_radians(self, deg):
        return deg * (np.pi / 180)

    def _angles_to_dict(self, angles):
        angles = {
            'min_angle_x': self._to_radians(angles[0]),
            'max_angle_x': self._to_radians(angles[1]),
            'min_angle_y': self._to_radians(angles[2]),
            'max_angle_y': self._to_radians(angles[3]),
            'min_angle_z': self._to_radians(angles[4]),
            'max_angle_z': self._to_radians(angles[5])
        }
        return angles

    def rot_matrix_x(self, theta):
        mat = np.zeros((3,3)).astype(np.float32)
        mat[0, 0] = 1.
        mat[1, 1] = np.cos(theta)
        mat[1, 2] = -np.sin(theta)
        mat[2, 1] = np.sin(theta)
        mat[2, 2] = np.cos(theta)
        return mat

    def rot_matrix_y(self, theta):
        mat = np.zeros((3,3)).astype(np.float32)
        mat[0, 0] = np.cos(theta)
        mat[0, 2] = np.sin(theta)
        mat[1, 1] = 1.
        mat[2, 0] = -np.sin(theta)
        mat[2, 2] = np.cos(theta)
        return mat

    def rot_matrix_z(self, theta):
        mat = np.zeros((3,3)).astype(np.float32)
        mat[0, 0] = np.cos(theta)
        mat[0, 1] = -np.sin(theta)
        mat[1, 0] = np.sin(theta)
        mat[1, 1] = np.cos(theta)
        mat[2, 2] = 1.
        return mat

    def pad_rotmat(self, theta):
        return np.hstack((theta, np.zeros((3,1))))

    def sample_angles(self,
                      bs,
                      min_angle_x,
                      max_angle_x,
                      min_angle_y,
                      max_angle_y,
                      min_angle_z,
                      max_angle_z):
        angles = []
        for i in range(bs):
            rnd_angles = [
                np.random.uniform(min_angle_x, max_angle_x),
                np.random.uniform(min_angle_y, max_angle_y),
                np.random.uniform(min_angle_z, max_angle_z),
            ]
            angles.append(rnd_angles)
        return np.asarray(angles)

    def get_theta(self, angles):
        bs = len(angles)
        theta = np.zeros((bs, 3, 4))

        angles_x = angles[:, 0]
        angles_y = angles[:, 1]
        angles_z = angles[:, 2]
        for i in range(bs):
            theta[i] = self.pad_rotmat(
                np.dot(np.dot(self.rot_matrix_z(angles_z[i]), self.rot_matrix_y(angles_y[i])),
                       self.rot_matrix_x(angles_x[i]))
            )

        return torch.from_numpy(theta).float()

    @staticmethod
    def _adain_module_3d(z_dim, out_ch):
        adain = nn.InstanceNorm3d(out_ch, affine=True)
        z_mlp = nn.Sequential(
            nn.Linear(z_dim, out_ch*2), # both var and mean
        )
        return adain, z_mlp

    @staticmethod
    def _adain_module_2d(z_dim, out_ch):
        adain = nn.InstanceNorm2d(out_ch, affine=True)
        z_mlp = nn.Linear(z_dim, out_ch*2)
        return adain, z_mlp

    def _rshp2d(self, z):
        return z.view(-1, z.size(1), 1, 1)

    def _rshp3d(self, z):
        return z.view(-1, z.size(1), 1, 1, 1)

    def _split(self, z):
        len_ = z.size(1)
        mean = z[:, 0:(len_//2)]
        var = F.softplus(z[:, (len_//2):])
        return mean, var

    def forward(self, z, thetas):
        bs = z.size(0)
        xstart = self.xstart.repeat((bs, 1, 1, 1, 1))  # (512, 4, 4, 4)

        h1 = self.adain_1(self.ups_3d(self.rb1(xstart)))  # (256, 8, 8, 8)
        z1_mean, z1_var = self._split(self._rshp3d(self.z_mlp1(z)))
        h1 = h1*z1_var + z1_mean

        h2 = self.adain_2(self.ups_3d(self.rb2(h1)))  # (128, 16, 16, 16)
        z2_mean, z2_var = self._split(self._rshp3d(self.z_mlp2(z)))
        h2 = h2*z2_var + z2_mean

        # Perform rotation
        grid = F.affine_grid(thetas, h2.size())
        h2_rotated = F.grid_sample(h2, grid, padding_mode='zeros')
        h4 = self.postproc(h2_rotated)  # (64, 16, 16, 16)

        # Projection unit. Concat depth and channels
        h4_proj = h4.view(-1, h4.size(1)*h4.size(2), h4.size(3), h4.size(4))  # (32*16, 16, 16) = (512, 16, 16)
        h4_proj = self.proj(h4_proj)  # (256, 16, 16)

        h5 = self.adain_3(self.ups_2d(self.rb1_2d(h4_proj)))  # (128, 32, 32)
        z3_mean, z3_var = self._split(self._rshp2d(self.z_mlp3(z)))
        h5 = h5*z3_var + z3_mean

        h6 = self.adain_4(self.ups_2d(self.rb2_2d(h5)))
        z4_mean, z4_var = self._split(self._rshp2d(self.z_mlp4(z)))
        h6 = h6*z4_var + z4_mean
        h_last = h6

        h_final = self.tanh(self.conv_final(h_last))  # (3, 32, 32)
        return h_final


class Discriminator(nn.Module):
    def __init__(self, n_features, z_dim):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.spec_norm = SpectralNorm
        self.pool = nn.AvgPool2d(4)

        self.base_disc = nn.Sequential(
            DiscInitResBlock(3, n_features),
            DiscResBlock(n_features, n_features * 2, stride=2),
            DiscResBlock(n_features * 2, n_features * 4, stride=2),
            DiscResBlock(n_features * 4, n_features * 8, stride=2),
        )
        self.d = DiscResBlock(n_features * 8, n_features * 8)
        self.q = DiscResBlock(n_features * 8, n_features * 8)

        # final fc layer init + norm
        self.fc = nn.Linear(n_features * 8, 1)
        nn.init.xavier_uniform(self.fc.weight.data, 1.)
        self.fc = self.spec_norm(self.fc)

        self.cls = nn.Linear(n_features * 8, z_dim + 3)
        nn.init.xavier_uniform(self.cls.weight.data, 1.)
        self.cls = self.spec_norm(self.cls)

    def forward(self, x):
        h = self.base_disc(x)
        h_d = self.pool(self.d(h))
        h_q = self.pool(self.q(h))
        h_d = h_d.view(-1, h_d.size(1))
        h_q = h_q.view(-1, h_q.size(1))

        pred_d = self.fc(h_d)
        pred_d = F.sigmoid(pred_d)  # if using sigmoid in final layer
        pred_zt = self.cls(h_q)
        pred_z = pred_zt[:, 0:self.z_dim]
        pred_t = pred_zt[:, self.z_dim:]
        return pred_d, pred_z, pred_t


########  util layers for discriminator and generator  ########
class GenResBlockNdim(nn.Module):
    def __init__(self, in_ch, out_ch, n_dims):
        super(GenResBlockNdim, self).__init__()
        ConvNd, InstanceNormNd = self._get_nd_blocks(n_dims=n_dims)
        self.relu = nn.LeakyReLU()

        self.conv1 = ConvNd(in_ch, out_ch, 3, 1, padding=1)
        self.conv2 = ConvNd(out_ch, out_ch, 3, 1, padding=1)
        self.bn = InstanceNormNd(in_ch)
        self.bn2 = InstanceNormNd(out_ch)

        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        bypass = []
        if in_ch != out_ch:
            bypass.append(ConvNd(in_ch, out_ch, 1, 1))
        self.bypass = nn.Sequential(*bypass)

    def _get_nd_blocks(self, n_dims):
        if n_dims == 2:
            ConvNd = nn.Conv2d
            InstanceNormNd = nn.InstanceNorm2d
        elif n_dims == 3:
            ConvNd = nn.Conv3d
            InstanceNormNd = nn.InstanceNorm3d
        else:
            raise NotImplementedError
        return ConvNd, InstanceNormNd

    def forward(self, inp):
        x = self.bn(inp)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + self.bypass(inp)


class DiscResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DiscResBlock, self).__init__()
        self.spec_norm = SpectralNorm

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                self.spec_norm(self.conv1),
                nn.ReLU(),
                self.spec_norm(self.conv2)
            )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                self.spec_norm(self.conv1),
                nn.ReLU(),
                self.spec_norm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
        self.bypass = nn.Sequential()
        if in_channels != out_channels:
            self.bypass = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform(self.bypass.weight.data, np.sqrt(2))
            self.bypass = self.spec_norm(self.bypass)
        if stride != 1:
            self.bypass = nn.Sequential(
                self.bypass,
                nn.AvgPool2d(2, stride=stride, padding=0)
            )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class DiscInitResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscInitResBlock, self).__init__()
        self.spec_norm = SpectralNorm

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

        self.model = nn.Sequential(
            self.spec_norm(self.conv1),
            nn.ReLU(),
            self.spec_norm(self.conv2),
            nn.AvgPool2d(2)
        )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            self.spec_norm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)
