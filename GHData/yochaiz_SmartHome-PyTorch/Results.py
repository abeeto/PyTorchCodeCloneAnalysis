import os
from shutil import move
import pandas as pd
from bokeh.io import output_file, save, show
from bokeh.plotting import figure
from bokeh.plotting.helpers import DEFAULT_PALETTE
from bokeh.layouts import column
from itertools import cycle


class ResultsLog(object):
    def __init__(self, path='results.csv', plot_path=None):
        self.path = path
        self.plot_path = plot_path or (self.path + '.html')
        self.results = None
        self.clear()

        # load existing results if path already exist
        if os.path.exists(self.path):
            self.results = pd.read_csv(self.path)

    def clear(self):
        self.figures = []

    def add(self, **kwargs):
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        if self.results is None:
            self.results = df
        else:
            self.results = self.results.append(df, ignore_index=True)

    def save(self, title='Training Results'):
        if len(self.figures) > 0:
            if os.path.isfile(self.plot_path):
                os.remove(self.plot_path)
            output_file(self.plot_path, title=title)
            plot = column(*self.figures)
            save(plot)
            self.clear()
        self.results.to_csv(self.path, index=False, index_label=False)

    def load(self, path=None):
        path = path or self.path
        if os.path.isfile(path):
            self.results = pd.read_csv(path)

    def show(self):
        if len(self.figures) > 0:
            plot = column(*self.figures)
            show(plot)

    def plot(self, x, y, title=None, xlabel=None, ylabel=None,
             width=800, height=400, colors=None, line_width=2,
             tools='pan,box_zoom,wheel_zoom,box_select,hover,reset,save'):
        xlabel = xlabel or x
        f = figure(title=title, tools=tools,
                   width=width, height=height,
                   x_axis_label=xlabel or x,
                   y_axis_label=ylabel or '')
        if colors is not None:
            colors = iter(colors)
        else:
            colors = cycle(DEFAULT_PALETTE)
        for yi in y:
            f.line(self.results[x], self.results[yi],
                   line_width=line_width,
                   line_color=next(colors), legend=yi)
        self.figures.append(f)

    def image(self, *kargs, **kwargs):
        fig = figure()
        fig.image(*kargs, **kwargs)
        self.figures.append(fig)


class Results:
    endedFolder = 'ended'

    def __init__(self, folderPath, folderName, epoch=1, nDecimalDigits=5):
        self.folderPath = folderPath
        self.folderName = folderName
        self.csvFullPath = '{}/results.csv'.format(folderPath)
        self.plotFullPath = '{}/results.html'.format(folderPath)

        self.results = ResultsLog(self.csvFullPath, self.plotFullPath)

        self.epoch = epoch
        self.nDecimalDigits = nDecimalDigits

    def addGameData(self, rewardVal, lossVal, plot=True, save=True):
        self.results.add(
            epoch=self.epoch,
            reward=round(rewardVal, self.nDecimalDigits),
            loss=round(lossVal, self.nDecimalDigits)
        )

        self.epoch += 1

        if plot:
            self.results.plot(x='epoch', y=['reward'], title='Reward', ylabel='reward')
            self.results.plot(x='epoch', y=['loss'], title='Loss', ylabel='loss')

        if save:
            self.results.save()

    def moveToEnded(self):
        newFolderPath = '{}/../{}/{}'.format(self.folderPath, self.endedFolder, self.folderName)
        move(self.folderPath, newFolderPath)
        self.folderPath = newFolderPath
