import glob

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download test data from open datasets.
from model import NeuralNetwork

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


for _saved_model in glob.glob("model-*.pth"):
    print(f"{_saved_model=}")
    model = NeuralNetwork()
    model.load_state_dict(torch.load(_saved_model))
    model.eval()
    total_samples = 0
    correct = 0
    for _test_sample in test_data:
        x = _test_sample[0]
        y = _test_sample[1]
        with torch.no_grad():
            total_samples += 1
            pred = model(x)
            predicted, actual = classes[pred[0].argmax(0)], classes[y]
            prediction_result = predicted == actual
            correct += prediction_result
            # if not prediction_result:
            #     print(f'Predicted: "{predicted}", Actual: "{actual}"')
    print(f"{100.0 * correct / total_samples} %")
