import numpy as np
import torch as torch


class Equation(object):
    """Base class for defining PDE related function."""

    def __init__(self, eqn_config):
        self.dim = eqn_config.dim
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.y_init = None

    def sample(self, num_sample):
        """Sample forward SDE."""
        raise NotImplementedError

    def f_tf(self, t, x, y, z):
        """Generator function in the PDE."""
        raise NotImplementedError

    def g_tf(self, t, x):
        """Terminal condition of the PDE."""
        raise NotImplementedError


class HJBLQ(Equation):
    """HJB equation in PNAS paper doi.org/10.1073/pnas.1718942115"""
    def __init__(self, eqn_config):
        super(HJBLQ, self).__init__(eqn_config)
        self.x_init = np.zeros(self.dim)
        self.sigma = np.sqrt(2.0)
        self.lambd = 1.0

    def sample(self, num_sample):
        dw_sample =(np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t)
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        x_sample = torch.from_numpy(x_sample)
        dw_sample =  torch.from_numpy(dw_sample)
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        return -self.lambd * torch.sum(torch.square(z), dim = 1, keepdim =True)

    def g_tf(self, t, x):
        return torch.log((1 + torch.sum(torch.square(x), dim =1, keepdim =True)) / 2)


class AllenCahn(Equation):
    """Allen-Cahn equation in PNAS paper doi.org/10.1073/pnas.1718942115"""
    def __init__(self, eqn_config):
        super(AllenCahn, self).__init__(eqn_config)
        self.x_init = np.zeros(self.dim)
        self.sigma = np.sqrt(2.0)

    def sample(self, num_sample):
        dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        return y - torch.pow(y, 3)

    def g_tf(self, t, x):
        return 0.5 / (1 + 0.2 * torch.sum(torch.square(x), dim = 1, keepdim=True))


class PricingDefaultRisk(Equation):
    """
    Nonlinear Black-Scholes equation with default risk in PNAS paper
    doi.org/10.1073/pnas.1718942115
    """
    def __init__(self, eqn_config):
        super(PricingDefaultRisk, self).__init__(eqn_config)
        self.x_init = np.ones(self.dim) * 100.0
        self.sigma = 0.2
        self.rate = 0.02   # interest rate R
        self.delta = 2.0 / 3
        self.gammah = 0.2
        self.gammal = 0.02
        self.mu_bar = 0.02
        self.vh = 50.0
        self.vl = 70.0
        self.slope = (self.gammah - self.gammal) / (self.vh - self.vl)

    def sample(self, num_sample):
        dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = (1 + self.mu_bar * self.delta_t) * x_sample[:, :, i] + (
                self.sigma * x_sample[:, :, i] * dw_sample[:, :, i])
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        piecewise_linear = torch.nn.relu(
            torch.nn.relu(y - self.vh) * self.slope + self.gammah - self.gammal) + self.gammal
        return (-(1 - self.delta) * piecewise_linear - self.rate) * y

    def g_tf(self, t, x):
        return torch.min(x, 1, keepdims=True)


class PricingDiffRate(Equation):
    """
    Nonlinear Black-Scholes equation with different interest rates for borrowing and lending
    in Section 4.4 of Comm. Math. Stat. paper doi.org/10.1007/s40304-017-0117-6
    """
    def __init__(self, eqn_config):
        super(PricingDiffRate, self).__init__(eqn_config)
        self.x_init = np.ones(self.dim) * 100
        self.sigma = 0.2
        self.mu_bar = 0.06
        self.rl = 0.04
        self.rb = 0.06
        self.alpha = 1.0 / self.dim

    def sample(self, num_sample):
        dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        factor = np.exp((self.mu_bar-(self.sigma**2)/2)*self.delta_t)
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = (factor * np.exp(self.sigma * dw_sample[:, :, i])) * x_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        temp = torch.sum(z, dim = 1, keepdim=True) / self.sigma
        return -self.rl * y - (self.mu_bar - self.rl) * temp + (
            (self.rb - self.rl) * torch.maximum(temp - y, 0))

    def g_tf(self, t, x):
        temp = torch.max(x, dim = 1, keepdim=True)
        return torch.maximum(temp - 120, 0) - 2 * torch.maximum(temp - 150, 0)


class BurgersType(Equation):
    """
    Multidimensional Burgers-type PDE in Section 4.5 of Comm. Math. Stat. paper
    doi.org/10.1007/s40304-017-0117-6
    """
    def __init__(self, eqn_config):
        super(BurgersType, self).__init__(eqn_config)
        self.x_init = np.zeros(self.dim)
        self.y_init = 1 - 1.0 / (1 + np.exp(0 + np.sum(self.x_init) / self.dim))
        self.sigma = self.dim + 0.0

    def sample(self, num_sample):
        dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        return (y - (2 + self.dim) / 2.0 / self.dim) * torch.sum(z, dim = 1, keepdim=True)

    def g_tf(self, t, x):
        return 1 - 1.0 / (1 + torch.exp(t + torch.sum(x, dim = 1, keepdim=True) / self.dim))


class QuadraticGradient(Equation):
    """
    An example PDE with quadratically growing derivatives in Section 4.6 of Comm. Math. Stat. paper
    doi.org/10.1007/s40304-017-0117-6
    """
    def __init__(self, eqn_config):
        super(QuadraticGradient, self).__init__(eqn_config)
        self.alpha = 0.4
        self.x_init = np.zeros(self.dim)
        base = self.total_time + np.sum(np.square(self.x_init) / self.dim)
        self.y_init = np.sin(np.power(base, self.alpha))

    def sample(self, num_sample):
        dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + dw_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        x_square = torch.sum(torch.square(x), dim = 1, keepdim=True)
        base = self.total_time - t + x_square / self.dim
        base_alpha = torch.pow(base, self.alpha)
        derivative = self.alpha * torch.pow(base, self.alpha - 1) * torch.cos(base_alpha)
        term1 = torch.reduce_sum(torch.square(z), dim= 1, keepdim=True)
        term2 = -4.0 * (derivative ** 2) * x_square / (self.dim ** 2)
        term3 = derivative
        term4 = -0.5 * (
            2.0 * derivative + 4.0 / (self.dim ** 2) * x_square * self.alpha * (
                (self.alpha - 1) * torch.pow(base, self.alpha - 2) * torch.cos(base_alpha) - (
                    self.alpha * torch.pow(base, 2 * self.alpha - 2) * torch.sin(base_alpha)
                    )
                )
            )
        return term1 + term2 + term3 + term4

    def g_tf(self, t, x):
        return torch.sin(
            torch.pow(torch.sum(torch.square(x), dim = 1, keepdim=True) / self.dim, self.alpha))


class ReactionDiffusion(Equation):
    """
    Time-dependent reaction-diffusion-type example PDE in Section 4.7 of Comm. Math. Stat. paper
    doi.org/10.1007/s40304-017-0117-6
    """
    def __init__(self, eqn_config):
        super(ReactionDiffusion, self).__init__(eqn_config)
        self._kappa = 0.6
        self.lambd = 1 / np.sqrt(self.dim)
        self.x_init = np.zeros(self.dim)
        self.y_init = 1 + self._kappa + np.sin(self.lambd * np.sum(self.x_init)) * np.exp(
            -self.lambd * self.lambd * self.dim * self.total_time / 2)

    def sample(self, num_sample):
        dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + dw_sample[:, :, i]
        return dw_sample, x_sample

    def f_tf(self, t, x, y, z):
        exp_term = torch.exp((self.lambd ** 2) * self.dim * (t - self.total_time) / 2)
        sin_term = torch.sin(self.lambd * torch.reduce_sum(x, dim = 1, keepdim=True))
        temp = y - self._kappa - 1 - sin_term * exp_term
        return torch.minimum(torch.tensor(1.0, dtype=torch.float64), torch.square(temp))

    def g_tf(self, t, x):
        return 1 + self._kappa + torch.sin(self.lambd * torch.reduce_sum(x, 1, keepdims=True))
