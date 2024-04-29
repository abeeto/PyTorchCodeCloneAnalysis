from src.Autoencoder import Autoencoder
import torchvision as tv
import torch as t
import os

NUM_EPOCHS = 1
SAVE_PATH = "saved_models/autoencoder_1_epoch.pt"
BATCH_SIZE = 1000


if __name__ == "__main__":
    if (not os.path.isdir("data")):
        os.mkdir("data/")
    if (not os.path.isdir("saved_models")):
        os.mkdir("saved_models/")
    
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    print(device)

    training_data = tv.datasets.MNIST("data/", train=True, download=True, transform=tv.transforms.ToTensor())
    training_data = t.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True) 
    test_data = tv.datasets.MNIST("data/", train=False, download=True, transform=tv.transforms.ToTensor())
    test_data = t.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    net = Autoencoder()
    if t.cuda.device_count() > 1:
        net = t.nn.DataParallel(net)
    net.to(device)

    for epoch in range(NUM_EPOCHS):
        for (x_batch, _) in training_data:
            x_batch = t.flatten(x_batch, 1).to(device)
            for count, x in enumerate(x_batch):
                err = net.backward(net(x), x)
                if not (count % 100):
                    print("Cross-Entropy error: {}".format(err))
    
    error_sum = 0
    num = 0
    for (x_batch, _) in test_data:
        num += x_batch.shape[0]
        x_batch = t.flatten(x_batch, 1).to(device)
        for x in x_batch:
            error_sum += net.evaluate(net(x), x)

    net.save(SAVE_PATH)
    print("Avg Cross-Entropy testing error: {}".format(error_sum/num))
    
