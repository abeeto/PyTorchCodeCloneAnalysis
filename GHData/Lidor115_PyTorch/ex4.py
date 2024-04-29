from torch import nn
from gcommand_loader import GCommandLoader
import torch


def dataTrain(data):
    # rnn = nn.RNNCell(n_inputs, n_neurons)
    rnn = nn.RNNCell(5, 5)
    
    for k, (input, label) in enumerate(data):
        print(input.size(), len(label))


    return

def main():
    data = GCommandLoader('./data/train')
    tensor = torch.utils.data.DataLoader(
        data, batch_size=100, shuffle=True,
        num_workers=20, pin_memory=True, sampler=None)
    dataTrain(tensor)
    return


if __name__== "__main__":
    main()