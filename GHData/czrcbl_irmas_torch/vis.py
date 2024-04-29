#%%
from librosa.display import specshow
from src.datasets import IRMAS
import numpy as np

trn_ds = IRMAS()

spec, label = trn_ds[0]

for i in range(10):
    spec, label = trn_ds[i]
    spec = spec.numpy()[0, :, :]
    print(np.max(spec), np.min(spec))

specshow(spec)

