import torch
from models import EfficientDet
from models.efficientnet import EfficientNet

if __name__ == '__main__':
    inputs = torch.randn(5, 3, 512, 512)

    # Test inference
    model = EfficientDet(num_classes=20, is_training=False)
    output = model(inputs)
    for out in output:
        print(out.size())
        # print(type(out))