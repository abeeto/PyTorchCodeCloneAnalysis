from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from .utils import tensors_from_pair

class SentencePairDataset(Dataset):
    
    def __init__(self, pairs, src_lang, dst_lang):
        self.pairs = [tensors_from_pair(pair, src_lang, dst_lang) 
                      for pair in pairs]
        self.src_lang = src_lang
        self.dst_lang = dst_lang
        self.length = len(self.pairs)
        self._pad_pairs(src_lang, dst_lang)

    def _pad_pairs(self, src_lang, dst_lang):
        src_pairs = list(map(lambda pair : pair[0], self.pairs))
        dst_pairs = list(map(lambda pair : pair[1], self.pairs))
        self.dict_pairs = {src_lang : pad_sequence(src_pairs, batch_first=True),
                           dst_lang : pad_sequence(dst_pairs, batch_first=True)} 

    def __getitem__(self, idx):
        return self.dict_pairs[self.src_lang][idx], self.dict_pairs[self.dst_lang][idx]

    def __len__(self):
        return self.length