import sys
import json
import torch
import argparse
from cnn import CNNModel
from sklearn.model_selection import ParameterGrid

HYPERPARAM_MODE = 'HYPERPARAM_MODE'
TRAINING_MODE = 'TRAINING_MODE'


def main():
    cuda = torch.cuda.is_available()
    if cuda:
        print('Device: {}'.format(torch.cuda.get_device_name(0)))

    json_params_file = sys.argv[1]
    with open(json_params_file, "r") as json_file:
        params = json.load(json_file)
        print(params)

        if HYPERPARAM_MODE == params['mode']:
            hyperparam_log_file = open('cnn_hyperparam_optimization.txt', 'w')
            hyperparam_log_file.write('batch_size,dims,lr,best_epoch,best_f1,last_f1\n')
            best_model_params = None
            best_model_f1 = -1
            base_params = params.copy()
            del base_params['mode']
            param_grid = ParameterGrid(base_params)
            for search_params in param_grid:
                print('Fitting model with params:',search_params)
                siamese_model =CNNModel(search_params, cuda)
                siamese_model.fit()
                hyperparam_log_file.write('{},{},{},{},{},{}\n'.format(
                    search_params['batch_size'],
                    search_params['dims'],
                    search_params['lr'],
                    siamese_model.best_epoch,
                    siamese_model.best_f1,
                    siamese_model.current_f1
                ))
                if siamese_model.best_f1 > best_model_f1:
                    best_model_f1 = siamese_model.best_f1
                    best_model_params = search_params
                    print('Best model params! F1:{}'.format(best_model_f1), best_model_params)
            hyperparam_log_file.close()
        elif params['mode'] == 'EVAL_MODE':
            siamese_model =CNNModel(params, cuda)
            siamese_model.eval()
        else:
            siamese_model =CNNModel(params, cuda)
            siamese_model.fit()




if __name__ == '__main__':
    main()
