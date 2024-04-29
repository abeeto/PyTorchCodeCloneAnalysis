from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import yaml

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # NOTE: do so before import torch

from lib.utils import load_graph_data
from model.pytorch.dcrnn_supervisor import DCRNNSupervisor


def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)
        supervisor_config['train']['epoch'] = args.epoch
        if args.log_dir:
            supervisor_config['train']['log_dir'] = args.log_dir
        supervisor_config['data']['seq_len'] = supervisor_config['model'].get('seq_len')
        supervisor_config['data']['horizon'] = supervisor_config['model'].get('horizon')

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)

        supervisor.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    parser.add_argument('--epoch', default=0, type=int)
    parser.add_argument('--log_dir', type=str)
    args = parser.parse_args()
    main(args)
