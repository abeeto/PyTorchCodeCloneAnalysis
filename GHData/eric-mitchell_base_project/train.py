import numpy as np
import torch
import torchvision as tv
import torch.nn.functional as F
from torchsummary import summary
 

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--name', type=str, default='BLANK_NAME')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--eps', type=float, default=1e-5)
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)

def run():
    # summary(model, (INPUT_SHAPE,))
    step = 0
    steps_per_epoch = 1000
    for epoch in range(args.epochs):
        for step in range(steps_per_epoch):
            # Do stuff
            
            if step % 250 == 0:
                # Log scalars
                # Guild will pick up anything following the 'KEY := VALUE' syntax
                #  as well as any files we save (images, checkpoints, etc)
                print(f'step := {step}')

                to_save = { }
                torch.save(to_save, 'checkpoint.pt')

            step += 1

if __name__ == '__main__':
    run()
