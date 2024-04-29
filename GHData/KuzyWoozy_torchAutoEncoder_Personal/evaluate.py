import torch as t
import torchvision as tv
from matplotlib import pyplot as plt
from src.Autoencoder import Autoencoder

SAVE_PATH = "saved_models/autoencoder_1_epoch.pt"


if __name__ == "__main__":
    
    test_data = tv.datasets.MNIST("data/", train=False, download=True, transform=tv.transforms.ToTensor())
    test_data = t.utils.data.DataLoader(test_data, batch_size=1)

    net = Autoencoder()
    net.load(SAVE_PATH)

    for (x, _) in test_data:
        plt.imshow(x[0][0].numpy())
        plt.show() 
        plt.imshow(net.predict(x.flatten()).view(28,28).numpy())
        plt.show()
