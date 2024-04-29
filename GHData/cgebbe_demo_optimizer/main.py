import optimize
import cost_func_torch
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.gridspec

# change default colors
import matplotlib as mpl
from cycler import cycler

mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrm')


def main():
    # define loss func
    # loss_func = lambda x: x.pow(2)
    # loss_func = cost_func_torch.create_line_func_with_saddle(slope_mid=0.1)
    # loss_func = cost_func_torch.create_line_func(1)
    loss_func = cost_func_torch.create_line_func_with_spike(0.5)

    # other parameters
    lst_x_start = [10]  # , -5, -10, -7]  # [-10, -7, -5, -3, -2.5, -1, 5, 10]
    lst_optimizer_name = ['sgd', 'adagrad', 'rmsprop', 'adam']  # , 'adam1', 'adam2', 'adam_default']
    nsteps_max = 5000

    for x_start in lst_x_start:
        print("Working on x_start={}".format(x_start))
        lst = []
        for optimname in lst_optimizer_name:
            # run optimization
            optimizer, optimizer_kwargs = get_optimizer(optimname)
            xs, losses = optimize.optimize(loss_func,
                                           optimizer=optimizer,
                                           optimizer_kwargs=optimizer_kwargs,
                                           x_start=x_start,
                                           nsteps_max=nsteps_max
                                           )
            lst.append((optimname, optimizer_kwargs, xs, losses))

        # plot and save
        title = "xstart_{:.3f}".format(x_start)
        fig, axes = plot(lst, loss_func)
        axes[1].set_xlim([0, nsteps_max])
        axes[0].set_xlim([-1, 11])
        if 0:
            axes[0].set_ylim([1.9,2.3])
            axes[0].set_xlim([0.8, 1.2])
            axes[1].set_xlim([1000, 1800])
        # fig.suptitle(title)
        fig.tight_layout()
        plt.show()
        fig.savefig("output/" + title + ".png")
        plt.close(fig)


def get_optimizer(name):
    lr = 0.005
    alpha = 0.99
    dct = {
        'adadelta': (torch.optim.Adadelta, {'lr': lr}),
        'adagrad': (torch.optim.Adagrad, {'lr': lr * 10}),  # if not optim.zero, *10 ?!
        'adam1': (torch.optim.Adam, {'lr': lr, 'betas': (0, alpha)}),  # default 0.9, 0.999
        'adam2': (torch.optim.Adam, {'lr': lr, 'betas': (alpha, alpha)}),  # default 0.9, 0.999
        'adam': (torch.optim.Adam, {'lr': lr, 'betas': (0.9, 0.999)}),  # default 0.9, 0.999
        'adamw': (torch.optim.AdamW, {'lr': lr}),
        'rmsprop': (torch.optim.RMSprop, {'lr': lr, 'alpha': alpha}),  # default 0.99
        'lbfgs': (torch.optim.LBFGS, {'lr': lr}),
        'sgd': (torch.optim.SGD, {'lr': lr}),  # * 50 in linear case!, since 0.02 line slope!
    }
    return dct[name]


def plot(lst, loss_func, xmin=-1, xmax=11):
    # create figure
    fig = plt.figure(figsize=(8, 2.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 5])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax1.get_shared_y_axes().join(ax1, ax2)

    # plot loss(x) - x
    xs = torch.tensor(np.linspace(xmin, xmax, 1000), requires_grad=False)
    losses = [loss_func(x) for x in xs]
    ax1.plot(losses, xs, 'k')
    ax1.set_xlabel('loss')
    ax1.set_ylabel('x')
    ax1.set_ylim([xmin, xmax])

    # plot x's along way
    linestyles = ['solid', 'dashdot', 'dashed', 'dotted', 'dotted', 'dotted']
    for cnt, (optimname, optimizer_kwargs, xs, losses) in enumerate(lst):
        time = range(len(xs))
        label = optimname + " with " + ",".join(["{}:{}".format(key, val) for key, val in optimizer_kwargs.items()])
        ax2.plot(time, xs,
                 linestyle=linestyles[cnt % len(linestyles)],
                 label=label,
                 )
    ax2.legend(loc='upper right')
    ax2.set_xlabel('time in optimization steps')
    ax2.set_ylabel('x')
    if False:
        ax2.tick_params(
            axis='y',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            left=False,  # ticks along the bottom edge are off
            right=False,  # ticks along the top edge are off
            labelleft=False,  # labels along the bottom edge are off
        )

    # make grid
    ax1.xaxis.grid(True)
    ax1.yaxis.grid(True)
    ax2.xaxis.grid(True)
    ax2.yaxis.grid(True)
    fig.tight_layout()

    return fig, [ax1, ax2]


if __name__ == '__main__':
    main()
