import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
from torch.utils.data import DataLoader

from torchsummary import summary

from datasets import *
from models import *
from parameters import args


if __name__ == '__main__':
    test_dic = make_datapath_dic('test')
    transform = ImageTransform(64)
    test_dataset = TripletDataset(test_dic, transform=transform, phase='test')
    batch_size = 32
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TripletNet().to(device)
    model.eval()
    model_weights = '0.14895377705494564.pth'
    model.load_state_dict(torch.load(model_weights))

    summary(model, (3, 64, 64))

    predicted_metrics = []
    test_labels = []
    with torch.no_grad():
        for i, (anchor, _, _, label) in enumerate(test_dataloader):
            metric = model(anchor).detach().cpu().numpy()
            metric = metric.reshape(metric.shape[0], metric.shape[1])
            predicted_metrics.append(metric)
            test_labels.append(label.detach().numpy())

    predicted_metrics = np.concatenate(predicted_metrics, 0)
    test_labels = np.concatenate(test_labels, 0)

    tSNE_metrics = TSNE(n_components=2, random_state=0).fit_transform(predicted_metrics)

    plt.scatter(tSNE_metrics[:, 0], tSNE_metrics[:, 1], c=test_labels)
    plt.colorbar()
    plt.savefig(model_weights + '.png')
    plt.show()
