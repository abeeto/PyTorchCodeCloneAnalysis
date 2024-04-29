import torch
from data_loader import FlickerDataset, filtered_collate_fn
import numpy as np
BATCH_SIZE = 16
NUM_PREPROCESS_WORKERS = 1
from PIL import Image


items = ['flicker_sample.csv']
dataset = FlickerDataset(items)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_PREPROCESS_WORKERS, drop_last=False, collate_fn=filtered_collate_fn)
try:
  for (ids, batch) in data_loader:
    try:
      print(ids)
    except Exception as e:
      print('Inner loop error:', e)
except Exception as e:
  print('Outer loop error:', e)
