import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io


def get_model():
    m = models.resnet18(pretrained=True)
    m.eval()
    return m


def image_to_tensor(image):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image))
    return my_transforms(image).unsqueeze(0)


def get_prediction_idx(image, model):
    tensor = image_to_tensor(image)
    outputs = model(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return predicted_idx
