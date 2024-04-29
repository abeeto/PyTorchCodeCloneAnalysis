import sys
import json
import torch
import argparse
from ff import create_ff_model

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

        if EVAL_MODE == params['mode']:
            ff_model = create_ff_model(params, cuda)
            ff_model.eval()
        elif EVAL_PERFORMANCE_MODE == params['mode']:
            ff_model = create_ff_model(params, cuda)
            ff_model.eval_performance()
        else:
            ff_model = create_ff_model(params, cuda)
            ff_model.fit()


if __name__ == '__main__':
    main()
