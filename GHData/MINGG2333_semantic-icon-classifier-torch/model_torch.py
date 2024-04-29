
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

class ICONNet(nn.Module):
    def __init__(
        self,
        embedding_size, num_classes, model_type='', siamese=False, conv_only=False
    ) -> None:
        super(ICONNet, self).__init__()
        
        if model_type == 'simple':
            print('model_type',model_type)

        if conv_only:
            print('conv_only',conv_only)

        self.model = nn.Sequential(
                nn.Conv2d(  1, 384, 3, padding='same',),
                nn.ELU(),
                nn.MaxPool2d(2),

                nn.Conv2d(384, 384, 1, padding='same',),
                nn.ELU(),
                nn.Conv2d(384, 384, 2, padding='same',),
                nn.ELU(),
                nn.Conv2d(384, 640, 2, padding='same',),
                nn.ELU(),
                nn.Conv2d(640, 640, 2, padding='same',),
                nn.ELU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.1),

                nn.Conv2d(640, 640, 1, padding='same',),
                nn.ELU(),
                nn.Conv2d(640, 768, 2, padding='same',),
                nn.ELU(),
                nn.Conv2d(768, 768, 2, padding='same',),
                nn.ELU(),
                nn.Conv2d(768, 768, 2, padding='same',),
                nn.ELU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.2),

                nn.Conv2d(768, 768, 1, padding='same',),
                nn.ELU(),
                nn.Conv2d(768, 896, 2, padding='same',),
                nn.ELU(),
                nn.Conv2d(896, 896, 2, padding='same',),
                nn.ELU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.3),

                nn.Conv2d(896, 896, 3, padding='same',),
                nn.ELU(),
                nn.Conv2d(896, 1024, 2, padding='same',),
                nn.ELU(),
                nn.Conv2d(1024, 1024, 2, padding='same',),
                nn.ELU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.4),

                nn.Conv2d(1024, 1024, 1, padding='same',),
                nn.ELU(),
                nn.Conv2d(1024, 1152, 2, padding='same',),
                nn.ELU(),
                nn.Dropout(0.5),

                nn.Flatten(),

                nn.Linear(1152, embedding_size,), # , name='embedding'
                nn.ELU(),
                nn.Linear(embedding_size, num_classes,), # , name='classification'
                # nn.Softmax(),
        )

    def forward(self, x):
        out = self.model(x)
        return out