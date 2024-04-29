import torch

# a = torch.randint(0,10,(2,3))
# print(a)
pos = (1,2)
def send(pos):
    print(pos)
for i in range(5):
    # print('Hana in my heart')
    send(pos)

# class MyDataModule(pl.DataModule):
#
#     def __init__(self):
#         ...
#
#     def prepare_data(self):
#         # called only on 1 GPU
#         download()
#         tokenize()
#         etc()
#
#     def train_dataloader(self):
#         # your train transforms
#         return DataLoader(YOUR_DATASET)
#
#     def val_dataloader(self):
#         # your val transforms
#         return DataLoader(YOUR_DATASET)
#
#     def test_dataloader(self):
#         # your test transforms
#         return DataLoader(YOUR_DATASET)