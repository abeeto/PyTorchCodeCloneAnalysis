import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch

from GMM_EM import GMM
from GMM_EM_log import GMM_log
from plot_clusters import PlotAllClusters


def train_model(x, k_components=3, model_class='GMM', epochs=20, threshold=3e-4):
    n, d = x.size(0), x.size(-1)

    if model_class == 'GMM':
        model = GMM(n=n, d=d, k_components=k_components)
    else:
        model = GMM_log(n=n, d=d, k_components=k_components)

    plot_model = PlotAllClusters(epochs)
    log_likelihood = torch.tensor([0])
    for epoch in range(epochs):
        prev_log_likelihood = log_likelihood

        mu, var, log_likelihood = model.train(x)
        delta = log_likelihood - prev_log_likelihood

        plot_model.plot_clusters(
            x=x,
            k_components=k_components,
            mu=mu,
            var=var,
            plot_num=epoch
        )
        # if delta <= threshold and epoch >= 3 :
        #     break

    plot_model.show_plot()

    return model

def predict_data(x, model):
    idx = model.predict(x)

    out = {}
    idx = idx.squeeze(1)
    for i in idx:
        if i.item() not in out:
            out[i.item()] = 0
        out[i.item()] += 1

    print(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--epoch',
        default=10,
        type=int
    )
    parser.add_argument(
        '--model_class',
        default='GMM',
        type=str
    )
    parser.add_argument(
        '--k',
        default=3,
        type=int
    )
    args = parser.parse_args()

    n, d = 150, 2
    # 生成樣本資料
    num1, mu1, var1 = 150, [0.5, 0.3], [1, 3]
    x1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    num2, mu2, var2 = 150, [5.5, 2.5], [2, 2]
    x2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    num3, mu3, var3 = 150, [1, 7], [6, 2]
    x3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    num4, mu4, var4 = 150, [-6, 1.2], [1, 4]
    x4 = np.random.multivariate_normal(mu4, np.diag(var4), num4)
    x = np.vstack((x1, x2, x3, x4))

    x = torch.from_numpy(x)

    plt.figure(figsize=(10, 8))
    plt.axis([-10, 10, -5, 10])
    plt.scatter(x1[:, 0], x1[:, 1], s=5)
    plt.scatter(x2[:, 0], x2[:, 1], s=5)
    plt.scatter(x3[:, 0], x3[:, 1], s=5)
    plt.scatter(x4[:, 0], x4[:, 1], s=5)
    plt.savefig('origin.png')

    x = x.unsqueeze(1)

    GMM_model = train_model(x, k_components=args.k, model_class=args.model_class, epochs=args.epoch)
    predict_data(x, GMM_model)
