from models.ref_denoise import DnCNN
from models.ref_denoise import MemNet
from models.ref_denoise import DHDN
from models.ref_denoise import FFDNet

from utils.data_loader import get_loader
# import experiments.trainer_ref_denoiser as trainer
import experiments.trainer_ref_denoiser_sidd as trainer
import torch
torch.manual_seed(0)

batch_size = 1
epoch_start = 0
num_epochs = 100
today = '2021.02.19'


"""
model = DnCNN()
train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
trainer.train(model, train_loader, test_loader, mode=f'ref_denoise_DNCNN', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=20, test_model_every=1, today=today)


model = MemNet(in_channels=3, channels=64, num_memblock=6, num_resblock=6)
train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
trainer.train(model, train_loader, test_loader, mode=f'ref_denoise_MemNet', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=20, test_model_every=1, today=today)


model = DHDN()
train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
trainer.train(model, train_loader, test_loader, mode=f'ref_denoise_DHDN', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=20, test_model_every=1, today=today)



model = FFDNet()
train_loader = get_loader(mode='train', batch_size=batch_size, height=192, width=192, scale_factor=4, augment=True)
test_loader = get_loader(mode='test', height=256, width=256, scale_factor=4)
trainer.train(model, train_loader, test_loader, mode=f'ref_denoise_FFDNet', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=20, test_model_every=1, today=today)
"""

today = '2021.03.05'


#model = DnCNN()
#train_loader = get_loader(data='SIDD', mode='train', batch_size=batch_size, height=192, width=192, scale_factor=1, augment=True)
#test_loader = get_loader(data='SIDD', mode='test', height=256, width=256, scale_factor=1)
#trainer.train(model, train_loader, test_loader, mode=f'ref_denoise_sidd_DNCNN', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=20, test_model_every=1, today=today)


#model = MemNet(in_channels=3, channels=64, num_memblock=6, num_resblock=6)
#train_loader = get_loader(data='SIDD', mode='train', batch_size=batch_size, height=192, width=192, scale_factor=1, augment=True)
#test_loader = get_loader(data='SIDD', mode='test', height=256, width=256, scale_factor=1)
#trainer.train(model, train_loader, test_loader, mode=f'ref_denoise_sidd_MemNet', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=20, test_model_every=1, today=today)


#model = DHDN()
#train_loader = get_loader(data='SIDD', mode='train', batch_size=batch_size, height=192, width=192, scale_factor=1, augment=True)
#test_loader = get_loader(data='SIDD', mode='test', height=256, width=256, scale_factor=1)
#trainer.train(model, train_loader, test_loader, mode=f'ref_denoise_sidd_DHDN', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=20, test_model_every=1, today=today)


model = FFDNet()
train_loader = get_loader(data='SIDD', mode='train', batch_size=batch_size, height=192, width=192, scale_factor=1, augment=True)
test_loader = get_loader(data='SIDD', mode='test', height=256, width=256, scale_factor=1)
trainer.train(model, train_loader, test_loader, mode=f'ref_denoise_sidd_FFDNet', epoch_start=epoch_start, num_epochs=num_epochs, save_model_every=20, test_model_every=1, today=today)

