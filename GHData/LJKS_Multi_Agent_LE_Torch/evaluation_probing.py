import argparse
import training
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
import scripts
import torch
import data
from marl_training import target_distractor_encode_data
def data_set(data, device, split):
    datapoints = [d[0] for d in data]
    targets = [d[1] for d in data]
    datapoints = np.concatenate(datapoints)
    targets = np.concatenate(targets)
    num_elems = (targets.shape[0])
    print(num_elems)
    p = np.random.permutation(num_elems)
    datapoints = datapoints[p]
    targets = targets[p]
    datapoints = np.array_split(datapoints, split)
    targets = np.array_split(targets, split)
    for x, t in zip(datapoints, targets):
        x = torch.from_numpy(x)
        t = torch.from_numpy(t)
        x = x.to(device)
        t = t.to(device)
        yield x, t





class Probe(torch.nn.Module):
    def __init__(self, input_size, classes):
        super(Probe, self).__init__()
        self.input_size = input_size
        self.classes = classes
        self.fc1 = torch.nn.Linear(self.input_size, self.classes)

    def forward(self, x):
        hidden = self.fc1(x)
        return hidden



if __name__ == '__main__':
    size = 2
    step = 100
    batch_size = 64
    (train_ds, test_ds), vocab = data.load_prepared_coco_data()
    device = torch.device('cuda')

    path = f'results/baseline_population_training/--num_senders_{size}--num_receivers_{size}--num_distractors_1--finetuning_lr_1e-05--batch_size_128--sender_sender_lstm64--receiver_receiver_lstm64'

    data_agg = [[] for _ in range(size)]
    for tscl_run_path in os.listdir(path):
        tscl_run_path = path + '/' + tscl_run_path + '/saves/finetune/'
        for i in range(size):
            sender_save_file = f'episode_{step}_sender_{i}.pt'
            sender = scripts.string_to_agent('sender_lstm64')
            sender.load_state_dict(torch.load(tscl_run_path + sender_save_file))
            sender.eval()
            sender.to(device)
            receivers = [scripts.string_to_agent('receiver_lstm64') for _ in range(size)]
            for receiver in receivers:
                receiver.load_state_dict(torch.load(tscl_run_path + f'episode_{step}_receiver_{i}.pt'))
                receiver.eval()
                receiver.to(device)
            data_loader_train = data.create_data_loader(
                train_ds, batch_size=batch_size, num_distractors=1, num_workers=4, device='gpu'
            )
            for all_features_batch, target_features_batch, target_captions_stack, target_idx_batch, ids_batch in data_loader_train:
                all_features_batch = all_features_batch.to(device=device)

                target_idx_batch = target_idx_batch.to(device=device)
                target_encoded_features = target_distractor_encode_data(all_features_batch, target_idx_batch, 2)  # before squeezing!
                target_idx_batch = torch.squeeze(target_idx_batch).to(device)


                seq, log_p = sender(target_encoded_features, seq_data=None, device=device)
                seq = seq.detach()  # technically not necessary but kind of nicer
                for j, receiver in enumerate(receivers):
                    features = receiver.extract_features(all_features_batch, seq)
                    features = features.detach().to('cpu').numpy()
                    data_agg[j].append([features.reshape((batch_size, -1)), np.asarray([i for _ in range(batch_size)])])
                del all_features_batch
                del target_idx_batch
                del target_encoded_features
                del seq

        print(tscl_run_path)
        num_probe_epochs = 10
        criterion = torch.nn.CrossEntropyLoss()
        for sender_idx in range(size):
            probe = Probe(64, size)
            probe.train()
            probe.to(device)
            optimizer = torch.optim.Adam(params=probe.parameters())

            for epoch in range(num_probe_epochs):
                data_gen = data_set(data_agg[sender_idx], device=device, split=400)
                for x,t in data_gen:
                    optimizer.zero_grad()
                    pred = probe(x)
                    loss = criterion(pred, t)
                    loss.backward()
                    optimizer.step()












