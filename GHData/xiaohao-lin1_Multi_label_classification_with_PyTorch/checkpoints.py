# Save the checkpoint
import torch
from torchvision import datasets, transforms, models


def save_checkpoint(model):
    #model.class_to_idx = training_dataset.class_to_idx

    checkpoint = {'arch': "resnet50",
                  #'class_to_idx': model.class_to_idx,
                  'model_state_dict': model.state_dict()
                  }

    torch.save(checkpoint, 'checkpoint.pth')

save_checkpoint(model)

#load the checkpoint
from collections import OrderedDict


# Function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    if checkpoint['arch'] == 'resnet50':

        model = models.resnet50(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False
    else:
        print("Architecture not recognized.")

    #model.class_to_idx = checkpoint['class_to_idx']

    fc = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(128, 6)
    )

    model.fc = fc

    model.load_state_dict(checkpoint['model_state_dict'])

    return model

# model = load_checkpoint('checkpoint.pth')
# print(model)