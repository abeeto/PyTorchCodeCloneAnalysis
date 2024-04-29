import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_factor_analysis import FA_Vectorised, FA_Numpy, FA_Standard
from pytorch_factor_analysis import generate_sample_data, plot
from configs import *

def main():
    # Training is faster on CPU
    device = torch.device('cpu')    

    # Set up the config file.
    cfg = {
        'METHOD': 'Vectorised',             # Vectorised is faster
        'PLOT_GRAPHS': True,
        'NUM_FEATURES': 2,
        'NUM_SAMPLES': 50,
        'NUM_FACTORS': 1,
        'NUM_ITERATIONS': 1000,
        'LOG_FREQ': 1,
        'RANDOM_SEED': 1
    }

    # Generate Sample Data for Plotting
    np.random.seed(cfg['RANDOM_SEED'])
    L = torch.Tensor([[4],[0.1]])
    S = torch.Tensor([[0.05,0],[0,0.02]])
    Z = torch.Tensor(np.random.normal(0, 1, [cfg['NUM_FACTORS'], cfg['NUM_SAMPLES']]))
    U = np.zeros([cfg['NUM_FEATURES'], cfg['NUM_SAMPLES']])
    for i in range(cfg['NUM_SAMPLES']):
        U[:,i] = np.random.normal(0, np.diag(S))  
    U = torch.Tensor(U)
    X = torch.Tensor(np.matmul(L,Z) + U)

    # Fit the EM algorithm
    fa = FA_Standard(cfg, device)
    L_pred, S_pred = fa.train_EM(X, L, S)       # Takes about 8 seconds to fit.

    # Train and plot Graphs
    if cfg['PLOT_GRAPHS']:
        plot(fa.metrics['L_error'], metric="Lambda_error")
        plot(fa.metrics['S_error'], metric="Psi_error")

    # Use Predictions to get X_preds
    U_preds = np.zeros([cfg['NUM_FEATURES'], cfg['NUM_SAMPLES']])
    for i in range(cfg['NUM_SAMPLES']):
        U_preds[:,i] = np.random.normal(0, np.diag(S_pred))
    X_preds = torch.Tensor(np.matmul(L_pred,Z) + torch.Tensor(U_preds).float())

    # Plot Figure using Predictions
    plt.figure(figsize=[16,9])
    plt.title("Example: Factor Analysis")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    x = np.linspace(-10, 10, 1000)
    plt.plot(x, x*(0.1/4), '-.k', label='Underlying Distribution (from Lambda)')
    plt.scatter(X[0,:], X[1,:], label='Sampled Inputs: X')
    plt.scatter(X_preds[0,:], X_preds[1,:], marker='x', label='Predictions: X_preds')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()

