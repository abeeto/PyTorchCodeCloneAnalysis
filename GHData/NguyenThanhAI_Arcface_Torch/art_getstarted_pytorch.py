"""
The script demonstrates a simple example of using ART with PyTorch. The example train a small model on the MNIST dataset
and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist

from backbones import get_model
from utils_fn import enumerate_images, load_images_and_labels_into_tensors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = r"D:\Face_Datasets\CelebA_Models\checkpoint.pth"
images_dir = r"D:\Face_Datasets\choose_train"
save_dir = r"D:\Face_Datasets\choose_train_art_images"
# Step 0: Define the neural network model, return logits instead of activation in forward method

class FaceModel(nn.Module):

    def __init__(self, model_name: str="r18", num_classes: int=10177):
        super().__init__()
        self.backbone = get_model(name=model_name)

        

        #in_features = self.backbone.features.out_features
        self.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    def forward(self, images):
        x = self.backbone(images)
        output = self.fc(x)

        return output



# Step 1: Load the MNIST dataset

images, labels = load_images_and_labels_into_tensors(images_dir=images_dir)

# Step 1a: Swap axes to PyTorch's NCHW format
print(images.shape, labels.shape)

# Step 2: Create the model

model = FaceModel(model_name="r18", num_classes=10)
model.load_state_dict(torch.load(weights, map_location=torch.device("cpu"))["weights"])
model.eval()
model.to(device=device)

# Step 2a: Define the loss function and the optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Step 3: Create the ART classifier

classifier = PyTorchClassifier(
    model=model,
    clip_values=(0, 1),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 112, 112),
    nb_classes=10,
)

# Step 4: Train the ART classifier


# Step 5: Evaluate the ART classifier on benign test examples

predictions = classifier.predict(images)
accuracy = np.sum(np.argmax(predictions, axis=1) == labels) / labels.shape[0]
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

# Step 6: Generate adversarial test examples
attack = FastGradientMethod(estimator=classifier, eps=0.2)
x_test_adv = attack.generate(x=images)

# Step 7: Evaluate the ART classifier on adversarial test examples

predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == labels) / labels.shape[0]
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))