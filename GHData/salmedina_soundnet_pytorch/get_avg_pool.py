import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str,
                        help='')
    parser.add_argument('-o', '--output_dir', type=str,
                        help='')
    parser.add_argument('-f', '--feats', type=str, default='conv7',
                        help='Feature selection: {conv7, y_obj, y_scns}')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    feats_path_list = list(Path(args.input_dir).glob('*.npy'))

    for feats_path in tqdm(feats_path_list):
        feats_path = Path(feats_path)
        feats_dict = np.load(feats_path, allow_pickle=True).item()
        sel_feats = feats_dict['conv7']
        # [N x C x S x 1] -> [C x S] -> C'
        avg_feats = sel_feats.squeeze().mean(1)

        # Saves on col-wise csv format
        np.savetxt(Path(args.output_dir, f'{feats_path.stem}.csv'),
                   avg_feats,
                   delimiter=';')
