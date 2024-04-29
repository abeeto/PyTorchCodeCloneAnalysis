"""
For each file, add a top-level comment addressing what the purpose of the
file does. If you follow the template, you likely should not have to change
much. However, if you add more files, remember to add this top-level comment.

Add an authors list below as shown. You can do this in a separate comment as
done in sklearn (https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/cluster/_mean_shift.py)

Authors:
    Rahul Yedida <rahul@ryedida.me>
"""
import logging
import warnings

from src.data import get_data
from src.experiments import Model, run_experiment
import os


# Global constants belong here. Add a comment describing what it is.
dataset_dic = {'ivy':     ['ivy-1.1.csv', 'ivy-1.4.csv', 'ivy-2.0.csv'],
               'lucene':  ['lucene-2.0.csv', 'lucene-2.2.csv', 'lucene-2.4.csv'],
               'poi':     ['poi-1.5.csv', 'poi-2.0.csv', 'poi-2.5.csv', 'poi-3.0.csv'],
               'synapse': ['synapse-1.0.csv', 'synapse-1.1.csv', 'synapse-1.2.csv'],
               'velocity': ['velocity-1.4.csv', 'velocity-1.5.csv', 'velocity-1.6.csv'],
               'camel': ['camel-1.0.csv', 'camel-1.2.csv', 'camel-1.4.csv', 'camel-1.6.csv'],
               'jedit': ['jedit-3.2.csv', 'jedit-4.0.csv', 'jedit-4.1.csv', 'jedit-4.2.csv', 'jedit-4.3.csv'],
               'log4j': ['log4j-1.0.csv', 'log4j-1.1.csv', 'log4j-1.2.csv'],
               'xalan': ['xalan-2.4.csv', 'xalan-2.5.csv', 'xalan-2.6.csv', 'xalan-2.7.csv'],
               'xerces': ['xerces-1.2.csv', 'xerces-1.3.csv', 'xerces-1.4.csv']
               }


if __name__ == '__main__':
    # Parse any args here. This demo code does not use args.

    # We prefer to use the logging library over print.
    # We do not have a preference for using a root logger like below or
    # using the static methods of the logging library.
    # Feel free to change the format.
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(filename)s:%(funcName)s:L%(lineno)d - %(message)s',
                        level=logging.DEBUG)

    warnings.filterwarnings('ignore')

    # It is YOUR job to ensure pre-conditions are satisfied.
    # For example, the code outputs to a log directory. It is
    # the resposibility of the code to create that directory.
    if 'ghost-log' not in os.listdir('.'):
        os.mkdir('./ghost-log')

    # Add comments regularly. Your code may speak for itself TO YOU,
    # but it is important that others understand your code anyway.
    for dataset in dataset_dic:
        logging.info(f'Running on dataset {dataset}')

        data_path = './data/'

        # Defer data fetching and preprocessing to a data module.
        train_ds, test_ds = get_data(data_path, dataset_dic[dataset])

        # This could probably be moved to src.experiments.run_experiment
        model = Model()

        name = dataset

        # The actual experiment should be abstracted away.
        # Do NOT put training code in this file.
        run_experiment(name, train_ds, test_ds, model)
