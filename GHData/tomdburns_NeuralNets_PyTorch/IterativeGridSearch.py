"""
Code that performs an iterative grid search of the data
"""


import os
import numpy as np
import pickle as pkl
from tqdm import tqdm
import itertools as it
from random import random


#TYPE   = ['Int', 'Int', 'Float']
#PARAMS = ['X', 'Y', 'Z']
#RANGES = [[0, 100], [0, 100], [0., 100.]]
#INTERV = 5


class Grid(object):
    """Grid optimization object"""

    def __init__(self, types, params, ranges, interv, logfile='OptimLog.pkl'):
        self.types    = types
        self.params   = params
        self.ranges   = ranges
        self.interval = interv
        self.grid     = {}
        self.testgrid = None
        self.best     = None
        self.last     = None
        self.next     = None
        self.done     = False
        self.logfile  = logfile
        self.generate_grid(self.types, self.params, self.ranges, self.interval)

    def generate_grid(self, type, params, ranges, interv):
        """Generates the grid"""
        arrays = []
        for i, param in enumerate(params):
            axis = np.linspace(ranges[i][0], ranges[i][1], interv)
            if 'INT' in type[i].upper():
                axis = np.array([int(v) for v in axis])
            elif 'LOG' in type[i].upper():
                axis = np.array([10**v for v in axis])
            arrays.append(axis)
        arrays = np.array(arrays)
        self.testgrid = np.array(list(it.product(*arrays)))

    def update_grid(self, results):
        """updates the grid data"""
        for i, _point in enumerate(self.testgrid):
            self.grid[self.assign_tag(_point)] = results[i]
        pkl.dump(self.grid, open(self.logfile, 'wb'))

    def assign_tag(self, point):
        """assigns a tag for a specific point"""
        tag = ''
        for i, val in enumerate(point):
            if self.types[i].upper() == 'INT':
                if i == 0:
                    tag += '%i' % val
                else:
                    tag += '-%i' % val
            if self.types[i].upper() == 'LOG':
                if i == 0:
                    tag += '%.8f' % np.log10(val)
                else:
                    tag += '-%.8f' % np.log10(val)
            else:
                if i == 0:
                    tag += '%.8f' % val
                else:
                    tag += '-%.8f' % val
        return tag

    def find_solution(self):
        """finds the best solution in the grid"""
        for point in self.grid:
            if self.best is None:
                self.best = (point, self.grid[point])
            elif self.grid[point] < self.best[1]:
                self.best = (point, self.grid[point])
        if self.last is None:
            self.last = self.best[0]
        elif self.last == self.best[0]:
            self.done = True
        else:
            self.last = self.best[0]

    def forward(self, N=1):
        """pushes the optimization to the next step"""
        self.next, _new = [], []
        # calculate ranges
        for i, axis in enumerate(self.ranges):
            _new.append((axis[1] - axis[0])/N)
        # get grid ranges for next step
        _best = [float(_i) for _i in self.best[0].split('-')]
        for _i, _ax in enumerate(_best):
            _low  = _ax - (_new[i]/2)
            _high = _ax + (_new[i]/2)
            if _low < self.ranges[i][0]:
                _low  = self.ranges[i][0]
                _high = _low + _new[i]
            elif _high > self.ranges[i][1]:
                _high = self.ranges[i][1]
                _low  = _high - _new[i]
            self.next.append([_low, _high])
        self.generate_grid(self.types, self.params, self.next, self.interval)

    def optimize(self, func, args, maxsteps=10):
        """Runs the optimization"""
        self.check_state()
        for _step in range(maxsteps):
            _N, _results = _step+1, []
            print('\nStarting Step:', _N)
            for _point in tqdm(self.testgrid):
                _tag = self.assign_tag(_point)
                _apoint = []
                for _arg in args:
                    _apoint.append(_arg)
                for _pnt in _point:
                    _apoint.append(_pnt)
                if _tag in self.grid:
                    _out = self.grid[_tag]
                else:
                    _out = func(*_apoint)
                _results.append(_out)
            self.update_grid(_results)
            self.find_solution()
            if self.done:
                print("\nSolution Found After %i Steps" % _N)
                break
            self.forward(N=_N+1)
        if not self.done:
            print("\nNo Solution Found After %i Steps" % maxsteps)
        _final = []
        for _i, _val in enumerate(self.best[0].split('-')):
            if self.types[_i].upper == 'INT':
                _final.append(int(_val))
            else:
                _final.append(float(_val))
        return _final, self.best[1]

    def check_state(self):
        """checks whether the optimization has already been performed"""
        if os.path.exists(self.logfile):
            self.grid = pkl.load(open(logfile, 'rb'))


def test_func(one, two, three):
    return random()


def main():
    """main"""
    grid = Grid()
    point, val = grid.optimize(test_func)


if __name__ in '__main__':
    main()
