import matplotlib.pyplot as plt
import pandas
import numpy as np

from fourier import fourier_series_coeff_numpy

d = 10

dataframe = pandas.read_csv('/home/nazar/PycharmProjects/ptbdb_normal.csv', engine='python').values

row = dataframe[1, 10:150]


from numpy import ones_like, cos, pi, sin, allclose
T = len(row) # any real number

def f(t):
    """example of periodic function in [0,T]"""
    n1, n2, n3 = 1., 4., 7.  # in Hz, or nondimensional for the matter.
    a0, a1, b4, a7 = 4., 2., -1., -3
    return a0 / 2 * ones_like(t) + a1 * cos(2 * pi * n1 * t / T) + b4 * sin(
        2 * pi * n2 * t / T) + a7 * cos(2 * pi * n3 * t / T)


def f1(t):

    return np.array([row[int(ti)] for ti in t])


N_chosen = 100
a0, a, b = fourier_series_coeff_numpy(f1, T, N_chosen)

t, dt = np.linspace(0,  4*T, 100, endpoint=False, retstep=True)

ff = a0 / 2 * ones_like(t)

for i in range(N_chosen):
    ff = ff + a[i] * cos(2 * pi * i * t / T) + b[i] * sin(2 * pi * i * t / T)


plt.plot(ff)
plt.show()

plt.plot(row)
plt.show()

