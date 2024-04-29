import os

import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
import numpy as np
from torch.utils.data import DataLoader
from collections import Counter
from torchtext.data.utils import get_tokenizer
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Sampler, Dataset

UNK, PAD, SOS = 0, 1, 2

class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            y = y.numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        n_batches = int(len(y) / batch_size)
        self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle)
        self.X = torch.randn(len(y),1).numpy()
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0,int(1e8),size=()).item()
        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return len(self.y)


tokenizer = get_tokenizer('basic_english')


def collate_batch(batch, SEQ_LEN, vocab):
    text_pipeline = lambda x: vocab(tokenizer(x))

    label_list, text_list = [], []
    for (_text,_label) in batch:
        label_list.append(_label)
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        processed_text = torch.cat((torch.tensor([SOS]),processed_text))
        if len(processed_text)>SEQ_LEN:
            processed_text = processed_text[:SEQ_LEN]
        else:
            pad = torch.tensor(PAD).repeat(SEQ_LEN-len(processed_text))
            processed_text = torch.cat((processed_text,pad))
        text_list.append(processed_text)

    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = torch.stack(text_list)
    return label_list, text_list

def get_imbalanced_loader(tr_dataset,ts_dataset,batch_size, SEQ_LEN, vocab):

    cb = lambda batch: collate_batch(batch, SEQ_LEN, vocab)

    sampler = StratifiedBatchSampler(tr_dataset.target, batch_size=batch_size)
    train_loader = DataLoader(tr_dataset, batch_sampler=sampler,
                                            shuffle=False, num_workers=int(8),collate_fn=cb)

    # train_loader = DataLoader(trainset,batch_size=BATCH_SIZE, num_workers=int(8),collate_fn=collate_batch)
    # batch = next(iter(dataloader))
    # batch[0]

    sampler = StratifiedBatchSampler(ts_dataset.target, batch_size=batch_size)
    test_loader = DataLoader(ts_dataset, batch_sampler=sampler,
                                            shuffle=False, num_workers=int(8),collate_fn=cb)

    print('훈련 데이터의 미니 배치의 개수 : {}'.format(len(train_loader)))
    print('테스트 데이터의 미니 배치의 개수 : {}'.format(len(test_loader)))

    return train_loader, test_loader

def get_oversampled_loader(tr_dataset,ts_dataset,batch_size, SEQ_LEN, vocab):

    cb = lambda batch: collate_batch(batch, SEQ_LEN, vocab)

    length = tr_dataset.__len__()

    labels = tr_dataset.target
    num_sample_per_class = Counter(labels)

    selected_list = []
    for i in range(0, length):
        _ = tr_dataset.__getitem__(i)
        label = _[1]
        if num_sample_per_class[label] > 0:
            selected_list.append(1 / num_sample_per_class[label])
            # selected_list.append(num_sample_per_class[label]/np.sum(list(num_sample_per_class.values())))
            # num_sample_per_class[label] -= 1

    sampler = WeightedRandomSampler(selected_list, len(selected_list))
    train_loader = DataLoader(tr_dataset, batch_size=batch_size,
                                 sampler=sampler, num_workers=0, drop_last=True,collate_fn=cb)

    # train_loader = DataLoader(trainset,batch_size=BATCH_SIZE, num_workers=int(8),collate_fn=collate_batch)
    # batch = next(iter(train_loader))
    # print(Counter(batch[1].numpy()))

    sampler = StratifiedBatchSampler(ts_dataset.target, batch_size=batch_size)
    test_loader = DataLoader(ts_dataset, batch_sampler=sampler,
                                            shuffle=False, num_workers=int(0),collate_fn=cb)

    print('훈련 데이터의 미니 배치의 개수 : {}'.format(len(train_loader)))
    print('테스트 데이터의 미니 배치의 개수 : {}'.format(len(test_loader)))

    return train_loader, test_loader

class MyDataset_origin(Dataset):

    def __init__(self, data, target, original_target):
        self.data = data
        self.target = target
        self.original_target = original_target

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        # return {"input":self.data[idx,:,:], 
        #         "label": self.target[idx]}
        return (self.data[idx],self.target[idx],self.original_target[idx])

class MyDataset(Dataset):

    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        # return {"input":self.data[idx,:,:], 
        #         "label": self.target[idx]}

        return (self.data[idx],self.target[idx])

def get_datset(dataset):
    return MyDataset(dataset.data,dataset.target)

def get_overlapped_datset(dataset,OR):
    idx = np.where(dataset.original_target==OR)[0]
    iidx = np.random.choice(idx,int(len(idx)*0.5),replace=True)
    additive_data = dataset.data[iidx]
    additive_target = np.zeros(additive_data.shape[0])

    ##############################################################################
    data = np.concatenate((additive_data,dataset.data))
    label = np.concatenate((additive_target,dataset.target))
    mydataset = MyDataset(data,label)
    ##############################################################################

    return mydataset
    
def save_checkpoint(acc, model, optim, epoch, SEED, LOGDIR, index=False):
    # Save checkpoint.
    print('Saving..')

    if isinstance(model, nn.DataParallel):
        model = model.module

    state = {
        'net': model.state_dict(),
        'optimizer': optim.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }

    if index:
        ckpt_name = 'ckpt_epoch' + str(epoch) + '_' + str(SEED) + '.t7'
    else:
        ckpt_name = 'ckpt_' + str(SEED) + '.t7'

    ckpt_path = os.path.join(LOGDIR, ckpt_name)
    torch.save(state, ckpt_path)