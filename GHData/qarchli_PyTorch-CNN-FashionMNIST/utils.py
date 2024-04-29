from itertools import product
from collections import namedtuple


class RunBuilder:
    """
    Class that builds runs from parameters' OrderedDict
    """
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs
