import numpy as np
import torch
from haudi import AudiSegmentationDataset

trainset = AudiSegmentationDataset("/mnt/sda/datasets/Audi/roadline", split="val", positional=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True)

totals = np.zeros((6,), dtype=np.int64)

for batch, (images, labels) in enumerate(trainloader):
    l = len(trainloader)
    print(f"{batch}/{l}")
    for index in range(len(totals)):
        totals[index] += np.count_nonzero(labels == index)

print(totals)
print(np.sum(totals))


# For TRAIN:
# [7345126014, 68308735, 61363929, 89002254, 7757468, 0]
# 7571558400

# For VAL:
# [524541516, 5031599, 4336765, 6211431, 550689, 0]
# 540672000