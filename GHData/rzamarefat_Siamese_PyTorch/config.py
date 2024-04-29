import torch

config = {
    "training_path": "/content/data/faces/training",
    "testing_path": "/content/data/faces/testing",
    "batch_size": 32,
    "lr": 0.0005,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}