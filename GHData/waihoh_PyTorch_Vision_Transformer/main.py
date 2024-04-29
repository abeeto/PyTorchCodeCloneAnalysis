import torch
from vision_transformer import VisionTransformer
from train import train
from prepare_dataset import data_loader

# Some parameters
EPOCHS = 2
LEARNING_RATE = 1e-3

device = torch.device('cpu')  # cuda:0

vision_transformer = VisionTransformer(image_size=32).to(device)
losses = train(vision_transformer, EPOCHS, LEARNING_RATE, data_loader, device)

print(losses)

