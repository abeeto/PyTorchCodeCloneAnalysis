import numpy as np
import torch
from gmm import GMM_torch
from sklearn import mixture
import time
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore")
if __name__ == "__main__":
    # Generate random sample, two components
    n_samples = 10000
    n_init = 3
    max_iter = 30
    n_components = 20
    np.random.seed(0)
    # C = np.array([[0.0, -0.1], [1.7, 0.4]])
    # X_numpy = np.r_[
    #     np.dot(np.random.randn(n_samples, 2), C),
    #     0.7 * np.random.randn(n_samples, 2) + np.array([-6, 3]),
    # ]           
    X_numpy = np.random.rand(n_samples,2)
    X_tensor_cpu = torch.from_numpy(X_numpy)
    X_tensor_gpu = torch.from_numpy(X_numpy).to('cuda')

    t_scikit = []
    t_cpu = []
    t_gpu = []

    # training
    for iter in range(100):
        sk_start_time = time.time()
        gmm_sklearn = mixture.GaussianMixture(n_components=n_components,n_init=n_init,max_iter=max_iter, covariance_type="full",tol=1e-20).fit(X_numpy)
        sk_end_time = time.time()
        t_scikit.append(sk_end_time - sk_start_time)

        torch_cpu_start_time = time.time()
        gmm_torch_cpu = GMM_torch(n_components=n_components, total_iter=max_iter,kmeans_iter=n_init)
        gmm_torch_cpu.fit(X_tensor_cpu)
        torch_cpu_end_time = time.time()
        t_cpu.append(torch_cpu_end_time - torch_cpu_start_time)

        torch_gpu_start_time = time.time()
        gmm_torch_gpu = GMM_torch(n_components=n_components, total_iter=max_iter,kmeans_iter=n_init)
        gmm_torch_gpu.fit(X_tensor_gpu)
        torch_gpu_end_time = time.time()
        t_gpu.append(torch_gpu_end_time - torch_gpu_start_time)

        print(f"\nloop = {iter}")
        print(f'scikit learn training time = \t\t{np.mean(t_scikit):.4f} +- {np.std(t_scikit):.4f} s')
        print(f'torch cpu learn training time = \t{np.mean(t_cpu):.4f} +- {np.std(t_cpu):.4f} s')
        print(f'torch gpu learn training time = \t{np.mean(t_gpu):.4f} +- {np.std(t_gpu):.4f} s')