from pyexpat import model
from models import vgg, resnet
from config import Config

def get_model(model_type,num_classes = 10):
    if model_type == "vgg11":
        return vgg.VGG11(Config.RESIZED_HEIGHT.value,Config.RESIZED_WIDTH.value)
    
    if model_type == "vgg13":
        return vgg.VGG13(Config.RESIZED_HEIGHT.value,Config.RESIZED_WIDTH.value)

    if model_type == "vgg16":
        return vgg.VGG16(Config.RESIZED_HEIGHT.value,Config.RESIZED_WIDTH.value)

    if model_type == "vgg19":
        return vgg.VGG19(Config.RESIZED_HEIGHT.value,Config.RESIZED_WIDTH.value)
    
    if model_type == 'resnet18':
        return resnet.ResNet18()

    if model_type == 'resnet34':
        return resnet.ResNet34()
    
    if model_type == 'resnet50':
        return resnet.ResNet50()
    
    if model_type == 'resnet101':
        return resnet.ResNet101()
    
    if model_type == 'resnet152':
        return resnet.ResNet152()