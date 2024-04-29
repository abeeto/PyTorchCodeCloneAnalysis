import torch
import argparse
import numpy as np
import pytorch_factor_analysis as pfa
from configs import configs

def main():
    parser = argparse.ArgumentParser(description='Accepts the config.')
    parser.add_argument('--config-idx', type=int,
                        help='Enter the index for your config file in the `configs` folder.')
    args = parser.parse_args()

    # Enter your config file below.
    cfg = configs[args.config_idx]

    device = torch.device('cpu')       # Training is faster on CPU
    # device = torch.device('cuda')      # Uncomment for training on GPU

    # Choose the EM implementation
    if cfg['METHOD'] == 'Vectorised':
        Z, L, S, U, X = pfa.generate_sample_data(cfg, toTensor=True, device=device)
        fa = pfa.FA_Vectorised(cfg, device)
    elif cfg['METHOD'] == 'Standard':
        Z, L, S, U, X = pfa.generate_sample_data(cfg, toTensor=True, device=device)
        fa = pfa.FA_Standard(cfg, device)
    elif cfg['METHOD'] == 'Numpy':
        Z, L, S, U, X = pfa.generate_sample_data(cfg, toTensor=False, device=device)
        fa = pfa.FA_Numpy(cfg, device)
    else:
        raise NotImplementedError()

    # Train and plot Graphs
    L_pred, S_pred = fa.train_EM(X, L, S)
    if cfg['PLOT_GRAPHS']:
        pfa.plot(fa.metrics['L_error'], metric="Lambda_error")
        pfa.plot(fa.metrics['S_error'], metric="Psi_error")

if __name__ == "__main__":
    main()

