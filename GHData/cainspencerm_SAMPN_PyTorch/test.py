import models, utils, dataset as ds  # local imports

import numpy as np
import torch
from torch.utils import data
import argparse


def main():
    parser = argparse.ArgumentParser(description='Test a model on a dataset.')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size.')
    parser.add_argument('--dataset', type=str, default='lipophilicity', help='Dataset type (lipophilicity or solubility).')
    parser.add_argument('--model', type=str, default='QSAR', help='Model type (QSAR or QSARPlus).')
    parser.add_argument('--weights', type=str, default='./checkpoint/lipophilicity/model_state_dict.pt', help='Path to the model weights.')
    args = parser.parse_args()

    # Prepare dataset and dataloaders.
    if args.dataset == 'lipophilicity':
        data_path = './data/LogP_moleculenet.csv'  # Lipophilicity
    elif args.dataset == 'solubility':
        data_path = './data/water_solubilityOCD.csv'  # Aqueous Solubility
    else:
        raise ValueError('Invalid dataset type: {}'.format(args.dataset))

    # Initialize the metrics.
    metric_funcs = list(utils.get_metrics(args.model).values())

    # Create the model.
    if args.model == 'QSAR':
        model = models.QSAR()
    elif args.model == 'QSARPlus':
        model = models.QSARPlus()
        args.batch_size = 1  # QSARPlus requires batch size of 1.
    else:
        raise ValueError('Invalid model type: {}'.format(args.model))

    test_set = ds.DGLDataset(data_path, split='test', shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, num_workers=16, shuffle=True)

    # Test the best model.
    model.load_state_dict(torch.load(args.weights))

    model.eval()

    scores = np.zeros(len(metric_funcs))

    with torch.no_grad():
        for smiles, labels in test_loader:

            preds = model(smiles)

            # Calculate metrics.
            for idx, metric_func in enumerate(metric_funcs):
                scores[idx] += metric_func(preds, labels)

    scores /= len(test_loader)

    for metric_name, score in zip(utils.get_metrics(args.model).keys(), scores):
        print('{:<10} {:.4f}'.format(metric_name, score))


if __name__ == '__main__':
    main()