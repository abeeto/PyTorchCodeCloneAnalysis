import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ztorch_simulation
import utils

plt_color_codes = 'bgrcmykw'

sim200 = ztorch_simulation.Simulation(std=2.0, steps=1, on_the_fly=True)

steps, centres, aff_groups, points, granularity = sim200.run_ekm()
vnf_groups = utils.group_points(points, aff_groups)

plot_ekm_results = True

if plot_ekm_results:
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    avail_colors = set(plt_color_codes)

    for gid in vnf_groups:
        group = vnf_groups[gid]
        if len(avail_colors) > 0:
            color = avail_colors.pop()
            ax.scatter(*group.T, c=color, alpha=0.3)

    ax.set_xlabel('CPU ($\mu$) [%]')
    ax.set_ylabel('Memory ($m$) [%]')
    ax.set_zlabel('Network ($\eta$) [%]')

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 100)
    plt.show()
