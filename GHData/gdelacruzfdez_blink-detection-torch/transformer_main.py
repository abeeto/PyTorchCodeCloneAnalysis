
import sys
import json
import torch
import argparse
from transformer import create_transformer_model
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
            hyperparam_log_file = open('transformer_hyperparam_optimization.txt', 'w')
            hyperparam_log_file.write('batch_size,dims,lr,sequence_len,transformer_hidden_units,best_epoch,best_f1,last_f1\n')
            best_model_params = None
            best_model_f1 = -1
            base_params = params.copy()
            del base_params['mode']
            param_grid = ParameterGrid(base_params)
            for search_params in param_grid:
                print('Fitting model with params:',search_params)
                transformer_model = create_transformer_model(search_params, cuda)
                transformer_model.fit()
                hyperparam_log_file.write('{},{},{},{},{},{},{},{}\n'.format(
                    search_params['batch_size'],
                    search_params['dims'],
                    search_params['lr'],
                    search_params['sequence_len'],
                    search_params['transformer_hidden_units'],
                    transformer_model.best_epoch,
                    transformer_model.best_f1,
                    transformer_model.current_f1
                ))
                if transformer_model.best_f1 > best_model_f1:
                    best_model_f1 = transformer_model.best_f1
                    best_model_params = search_params
                    print('Best model params! F1:{}'.format(best_model_f1), best_model_params)
            hyperparam_log_file.close()
        elif EVAL_MODE == params['mode']:
            transformer_model = create_transformer_model(params,cuda)
            transformer_model.eval()
        elif EVAL_PERFORMANCE_MODE == params['mode']:
            transformer_model = create_transformer_model(params,cuda)
            transformer_model.eval_performance()
        else:
            transformer_model = create_transformer_model(params, cuda)
            transformer_model.fit()


if __name__ == '__main__':
    main()
