import torch
import torch.nn.functional as F
import data
import utils
import math
from tqdm import tqdm
import numpy as np
from positional_encodings import PositionalEncoding1D
START_OF_SEQUENCE_TOKEN = 0

def pretrain_cbow_embedding(n, embedding_size, epochs, batch_size=128, lr=0.0001, vocab_size=2000, device='cpu'):
    cbow_average_layer = utils.cbow_average_module(dim=1)
    embedding_model = torch.nn.Sequential(torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size), cbow_average_layer, torch.nn.Linear(in_features=embedding_size, out_features=vocab_size))
    embedding_model.to(device=device)
    optimizer = torch.optim.SGD(embedding_model.parameters(), lr=lr, momentum=0.0)
    optimizer.zero_grad()
    print(f'pretraining word embedding')
    (train_ds, test_ds), vocab = data.load_prepared_coco_data()

    dataset = data.create_ngram_loader(train_ds, batch_size=batch_size, n=n, num_workers=4)

    target_token_idx = math.ceil(n/2) #round is not good enough with float precision, rounds down e.g. 5/2

    cce = torch.nn.CrossEntropyLoss()

    for epoch in tqdm(range(epochs)):
        epoch_loss = []
        for elem in dataset:
            target = elem[:,target_token_idx]
            pre_target_elems = elem[:,0:(target_token_idx-1)]
            post_target_elems = elem[:,target_token_idx:]
            inputs = torch.cat((pre_target_elems, post_target_elems), dim=1)

            target = target.to(device=device)
            inputs = inputs.to(device=device)

            optimizer.zero_grad()

            pred = embedding_model(inputs)
            loss = cce(pred, target)

            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.detach().to(device='cpu').numpy())
        print(f'epoch {epoch} loss: {np.mean(epoch_loss)}')


if __name__ == '__main__':
    device = torch.device('cuda:0')
    pretrain_cbow_embedding(5, 128, 100, batch_size=1024, lr=0.0001, vocab_size=2000, device=device)






