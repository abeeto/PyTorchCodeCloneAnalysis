
import sys
import json
import torch
import argparse
from lstm_cnn import create_lstm_model
from sklearn.model_selection import ParameterGrid

HYPERPARAM_MODE = 'HYPERPARAM_MODE'
TRAINING_MODE = 'TRAINING_MODE'
EVAL_MODE = 'EVAL_MODE'
EVAL_PERFORMANCE_MODE = 'EVAL_PERFORMANCE_MODE'


def main():
    cuda = torch.cuda.is_available()
    if cuda:
        print('Device: {}'.format(torch.cuda.get_device_name(0)))

    json_params_file = sys.argv[1]
    with open(json_params_file, "r") as json_file:
        params = json.load(json_file)
        print(params)

        if HYPERPARAM_MODE == params['mode']:
            hyperparam_log_file = open('lstm_hyperparam_optimization.txt', 'w')
            hyperparam_log_file.write('batch_size,dims,lr,sequence_len,lstm_hidden_units,best_epoch,best_f1,last_f1\n')
            best_model_params = None
            best_model_f1 = -1
            base_params = params.copy()
            del base_params['mode']
            param_grid = ParameterGrid(base_params)
            for search_params in param_grid:
                print('Fitting model with params:',search_params)
                lstm_model = create_lstm_model(search_params, cuda)
                lstm_model.fit()
                hyperparam_log_file.write('{},{},{},{},{},{},{},{}\n'.format(
                    search_params['batch_size'],
                    search_params['dims'],
                    search_params['lr'],
                    search_params['sequence_len'],
                    search_params['lstm_hidden_units'],
                    lstm_model.best_epoch,
                    lstm_model.best_f1,
                    lstm_model.current_f1
                ))
                if lstm_model.best_f1 > best_model_f1:
                    best_model_f1 = lstm_model.best_f1
                    best_model_params = search_params
                    print('Best model params! F1:{}'.format(best_model_f1), best_model_params)
            hyperparam_log_file.close()
        elif EVAL_MODE == params['mode']:
            lstm_model = create_lstm_model(params,cuda)
            lstm_model.eval()
        elif EVAL_PERFORMANCE_MODE == params['mode']:
            lstm_model = create_lstm_model(params,cuda)
            lstm_model.eval_performance()
        else:
            lstm_model = create_lstm_model(params, cuda)
            lstm_model.fit()


if __name__ == '__main__':
    main()
