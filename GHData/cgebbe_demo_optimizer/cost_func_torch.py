# import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt


def gauss(x, mu, sig):
    scalor = 1  # / (sig * np.sqrt(2 * np.pi))
    f = scalor * torch.exp(-0.5 * (x - mu) ** 2 / (sig ** 2))
    return f


def create_line_func(slope=1.0):
    def line(xs):
        ys = torch.abs(xs * slope)
        return ys

    return line


def create_line_func_with_spike(slope=1.0):
    def line(x):
        y = torch.abs(x * slope)
        if 2 < x and x < 2.2:
            y += 0.1 * gauss(x, 2.1, 0.03)
        return y

    return line


def create_line_func_with_saddle(slope=1.0, x_saddle=5, width_saddle=1, slope_mid=0.1):
    """ line with three sections: lo, mid, hi
    General idea: (y-y0) = (x-x0)*slope
    """

    def line(x):
        # define the two intersection points: "lo" for lo-mid and "hi" for mid-hi
        x_lo = x_saddle
        x_hi = x_saddle + width_saddle
        y_lo = x_lo * slope
        y_hi = (x_hi - x_lo) * slope_mid + y_lo

        if x < x_lo:
            y = x * slope
        elif x < x_hi:
            y = (x - x_lo) * slope_mid + y_lo
        else:  # x >= x_hi
            y = (x - x_hi) * slope + y_hi
        return torch.abs(y)

    return line


def cost(x):
    if x < 0:
        return 1 - gauss(x, 0, 5) - 0.19 * gauss(x, -5, 1)
    else:
        return create_line_func(0.02)(x)


if __name__ == '__main__':
    x = torch.linspace(-1, 11, 100)
    cost = create_line_func_with_saddle()
    y = [cost(xx) for xx in x]
    plt.plot(x, y)
    plt.show()
    print("finished")
