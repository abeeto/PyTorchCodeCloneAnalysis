import torch
import torch.nn as nn
from torch import backends
from torch.utils.tensorboard import SummaryWriter

import data_setup
import engine
import model_builder
import utils

INPUT_WIDTH = 64
NUM_CLASSES = 100
NUM_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
LOAD_MODEL = False

# CECKPOINT_PATH = "/content/gdrive/" + "MyDrive/model.pt"
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Load datasets
train_loader, valid_loader = data_setup.data_loader(data_dir='./data',
                                                    batch_size=BATCH_SIZE,
                                                    input_width=INPUT_WIDTH)

test_loader = data_setup.data_loader(data_dir='./data',
                                     batch_size=BATCH_SIZE,
                                     input_width=INPUT_WIDTH,
                                     test=True)

# Load model
if LOAD_MODEL:
    pass
else:
    model = model_builder.VGG_Net(num_classes=NUM_CLASSES,
                                  net_type="VGG-16",
                                  input_width=INPUT_WIDTH).to(device)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

writer = SummaryWriter()
engine.train(model=model,
             train_loader=train_loader,
             valid_loader=valid_loader,
             writer=writer,
             optimizer=optimizer,
             loss_fn=loss_fn,
             epochs=NUM_EPOCHS,
             device=device)
