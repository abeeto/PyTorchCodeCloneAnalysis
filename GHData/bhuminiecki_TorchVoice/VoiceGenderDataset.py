import os
import torch
from torch.utils.data import Dataset
from loader import load_file
from librosa import load, piptrack
import numpy as np
from scipy.signal.windows import blackmanharris


class VoiceGenderDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.entries = []
        self.labels = []
        for cat in os.walk(root_dir):
            for file in cat[2]:
                self.entries.append(os.path.join(cat[0], file))
                self.labels.append(cat[0][-1])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        target = self.labels[idx]

        target = torch.tensor(1 if (target == "k") else 0).to(torch.long)

        y, sr = load(self.entries[idx])
        pitches, magnitudes = piptrack(y=y, sr=sr, fmin=75, fmax=275)
        q25 = 0.0
        iqr = 0.0
        freq = 0.0
        if len(pitches[np.nonzero(pitches)]) > 0:
            q25 = np.percentile(pitches[np.nonzero(pitches)], 25)
            iqr = np.percentile(pitches[np.nonzero(pitches)], 75) - q25

            windowed = y * blackmanharris(len(y))
            median = np.argmax(abs(np.fft.rfft(windowed)))
            freq = median / len(windowed)

        out = torch.tensor([q25, iqr, freq])
        out.resize_(1, 3)

        return out, target

    def print_entries(self):
        print(self.entries)
        print(self.labels)


'''vgd = VoiceGenderDataset("data/train", transform=None)

for x in range(80):
    tensor = vgd.__getitem__(x)[0]
#    print((tensor).size())
    print(tensor)
#    print("############")
'''
