import os
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
from torchvision.transforms import ToTensor, Compose, Resize, ToPILImage


def load_checkpoint(fil_epath):
    checkpoint = torch.load(fil_epath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


def predict(params):

    image_transform = ToPILImage()
    input_transform = Compose([
        Resize((params["resize_width"], params["resize_height"])),
        ToTensor(),
    ])
    predict_transform = Compose([
        Resize((params["img_height"], params["img_width"]))
    ])

    model = load_checkpoint(params["model_dir"])
    test_dir = params["test_dir"]
    test_img = Image.open(test_dir).convert('RGB')

    image = input_transform(test_img)
    output = model(Variable(image, volatile=True).unsqueeze(0))
    label_raw = output[0].data.max(0)[1].unsqueeze(0)
    label_np = np.transpose(label_raw.cpu().detach().numpy(), (1, 2, 0))
    label_normalized = (((label_np - label_np.min()) / (label_np.max() - label_np.min())) * 255).astype(np.uint8)
    prediction = predict_transform(image_transform(label_normalized))
    prediction.save('predictions.jpg')


if __name__ == '__main__':
    params = {
        "test_dir": os.path.join(os.getcwd(), 'test_images/rgb.png'),
        "model_dir": os.path.join(os.getcwd(), 'weights/checkpoint.pth'),
        "img_height": 2022,
        "img_width": 1608,
        "resize_width": 256,
        "resize_height": 256
    }
    predict(params)