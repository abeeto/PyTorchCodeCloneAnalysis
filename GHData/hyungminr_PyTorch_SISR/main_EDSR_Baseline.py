import trainer
from models.EDSR import EDSR
from utils.data_loader import get_loader
import torch

torch.manual_seed(0)

scale_factor = 4

if scale_factor == 4:
    train_loader = get_loader(mode='train', batch_size=16, height=192, width=192, scale_factor=4, augment=True)
    test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
elif scale_factor == 2:
    train_loader = get_loader(mode='train', batch_size=16, augment=True)
    test_loader = get_loader(mode='test')

model = EDSR(scale=scale_factor)

trainer.train(model, train_loader, test_loader, mode='EDSR_x4_Baseline')
