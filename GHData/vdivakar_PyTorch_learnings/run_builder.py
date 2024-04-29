from itertools import product
from collections import namedtuple
from collections import OrderedDict

class RunBuilder():
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Yo', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs

my_params = OrderedDict(
    lr = [0.01, 0.001],
    batch_size = [1000, 10000]
)

all_runs = RunBuilder.get_runs(my_params)
for run in all_runs:
    print(run, run.lr, run.batch_size)

'''Training Loop will now look like this:- '''
for run in RunBuilder.get_runs(my_params):
    comment = f'-{run}'
    # ...