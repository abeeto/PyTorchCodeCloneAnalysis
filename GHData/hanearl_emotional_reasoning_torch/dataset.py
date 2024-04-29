from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pickle
import numpy as np


class SentenceDataset(Dataset):
    def __init__(self, data_path, bert_model_name):
        with open(data_path, 'rb') as f:
            sentences, labels = pickle.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        encoding = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)

        self.input_ids = encoding['input_ids']
        self.attention_mask = encoding['attention_mask']
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'inputs': self.input_ids[idx], 'mask': self.attention_mask[idx], 'labels': self.labels[idx]}


# sentence_dataset = SentenceDataset('data/val_set.pkl')
# dataloader = DataLoader(sentence_dataset, batch_size=4, shuffle=True, num_workers=4)
# print(len(sentence_dataset))
# print(len(dataloader.dataset))
#
# for i_batch, sample_batched in enumerate(dataloader):
#     print(i_batch, sample_batched['inputs'], sample_batched['mask'], sample_batched['labels'])
#     break
