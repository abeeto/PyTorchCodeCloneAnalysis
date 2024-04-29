from torchfusion.learners import *
import torch
from torchvision.models.squeezenet import squeezenet1_1
from torchfusion.utils import load_image,decode_imagenet
from torchfusion.datasets import *
INFER_FOLDER  = r"./images"
MODEL_PATH = r"squeezenet.pth"

#load images
infer_set = pathimages_loader([INFER_FOLDER],size=224,recursive=False)

#init squeezenet
net = squeezenet1_1()

#init learner and load squeezenet model
learner = StandardLearner(net)
learner.load_model(MODEL_PATH)

#function for predicting from a loader
def predict_loader(data_loader):
    predictions = learner.predict(data_loader)
    for pred in predictions:
        pred = torch.softmax(pred,0)
        class_index = torch.argmax(pred)
        class_name = decode_imagenet(class_index)
        confidence = torch.max(pred)
        print("Prediction: {} , Confidence: {} ".format(class_name, confidence))

#function for predict a single image
def predict_image(image_path):
    img = load_image(image_path,target_size=224,mean=0.5,std=0.5)
    img = img.unsqueeze(0)
    pred = learner.predict(img)
    pred = torch.softmax(pred,0)
    class_index = torch.argmax(pred)
    class_name = decode_imagenet(class_index)
    confidence = torch.max(pred)
    print("Prediction: {} , Confidence: {} ".format(class_name, confidence))


if __name__ == "__main__":
    predict_loader(infer_set)
    predict_image(r"sample.jpg")
