import os
from PIL import Image
import numpy
import torch
from mnist_model import Model

with torch.no_grad():
    #Choose folder with numbers 1/ You can change this to any numbers folder
    path = "C:/Users/Samuel/PycharmProjects/pythonProject/MNIST/MNIST - JPG - training/1/"
    f = os.listdir(path)
    #Choose random number 1 photo from floder, in this case we have 569th image of 1 from floder
    x = Image.open(os.path.join(path, f[569]))

    #Prepare image for model
    img = numpy.array(x)
    img = numpy.expand_dims(img, axis=0)
    g = (img / 255.0).astype(numpy.float32)
    g = numpy.expand_dims(g, axis=0)
    y = torch.from_numpy(g)

    #Load model
    PATH = "./MNIST-MY.pth"
    model =Model()
    model.load_state_dict(torch.load(PATH))

    #Push image to model
    y_pred = model(y)

    #Get prediction
    y_pred = torch.argmax(y_pred, dim=1)
    a =  y_pred.detach().numpy()[0]

    print("Predicted number is - " , a)

