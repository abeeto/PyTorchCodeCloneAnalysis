from __future__ import print_function

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from VoiceGenderDataset import VoiceGenderDataset
from loader import load_file
import sys


if len(sys.argv) > 1:
    filename = sys.argv[1]

    model = torch.load("model/model_epoch_10.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    classes = ["M", "K"]

    input = load_file(filename)
    input.resize_(1, 1, 3)

    prediction = model(input.to(device))

    predicted_class = prediction.argmax().item()
    print(classes[predicted_class])
else:
    print("Did not specify input file.")

