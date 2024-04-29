import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib import animation
from random import choices, random
import mppi


class Simulator():
    """
    A simple simulator.
    """
    def __init__(self,
                 problem):
        self.problem = problem
        self.blit = True
        self.num_steps = 100
        self.interval = 50
        self.repeat = True
        self.fig, self.ax = plt.subplots(1,2, figsize=(15, 8))
        self.fig.tight_layout()
        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.ax[0].axis('equal')
        self.ax[0].set(xlim = (-1.1, 1.1), ylim=(-1.1, 1.1))
        self.ax[1].set(xlim = (0, 10), ylim=(0, 1500.0))
        self.mouse_x = 0.0
        self.mouse_y = 0.0
        
    def onclick(self, event):
        self.problem.onclick(event.xdata, event.ydata)

    def on_mouse_move(self, event):
        self.mouse_x, self.mouse_y = event.xdata, event.ydata

    def init_func(self):
        buffer = []
        artists = self.problem.init_func()
        for patch in artists.patches:
            buffer.append(self.ax[0].add_patch(patch))
        for line in artists.plots:
            buffer.append(self.ax[1].add_line( line))
        self.ax[0].set_title("MPPI demo")
        self.ax[1].set_title("Total cost history")
        return buffer

    def callback_func(self, frame_ind):
        buffer = []
        artists = self.problem.callback_func()
        for patch in artists.patches:
            buffer.append(self.ax[0].add_patch(patch))
        for line in artists.plots:
            buffer.append(self.ax[1].add_line( line))
        # draw hair
        horizontal_line = self.ax[0].axhline(color='k', lw=0.8, ls='--')
        vertical_line = self.ax[0].axvline(color='k', lw=0.8, ls='--')
        text = self.ax[0].text(0.72, 0.9, '', transform=self.ax[0].transAxes)
        horizontal_line.set_ydata(self.mouse_y)
        vertical_line.set_xdata(self.mouse_x)
        buffer.append(horizontal_line)
        buffer.append(vertical_line)
        return buffer

    def run(self):
        anim = animation.FuncAnimation(self.fig,
                                       self.callback_func,
                                       init_func=self.init_func,
                                       frames=self.num_steps,
                                       interval=self.interval,
                                       blit=self.blit,
                                       repeat=self.repeat)


        plt.show()
