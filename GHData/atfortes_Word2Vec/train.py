import numpy as np
from tqdm.auto import tqdm
from utils import calc_spearman
from constants import TEST_FILE_PATH


def train(model, dataset, dataloader, optimizer, epochs, device):
    loss_hist = []
    for epoch in range(epochs):
        epoch_loss_hist = []
        for _, batch in enumerate(tqdm(dataloader)):
            center, context, neg = batch
            center = center.to(device)
            context = context.to(device)
            neg = neg.to(device)

            optimizer.zero_grad()
            loss = model(center, context, neg)
            epoch_loss_hist.append(loss.item())
            loss.backward()
            optimizer.step()

        loss_hist.append(np.mean(epoch_loss_hist))
        print(f'epoch: {epoch+1} \t avg_loss={loss_hist[-1]}')
        print(f'epoch: {epoch+1} \t spearman_corr={calc_spearman(TEST_FILE_PATH, model, dataset)}')
