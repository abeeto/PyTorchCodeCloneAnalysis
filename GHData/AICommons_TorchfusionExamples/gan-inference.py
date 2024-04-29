from torchfusion.gan.learners import *
from torchfusion.gan.applications import StandardGenerator
import torch.cuda as cuda
import torch.nn as nn
from torchvision.utils import save_image
import torch
from torch.distributions import Normal


G = StandardGenerator(output_size=(1,32,32),latent_size=128,num_classes=10)

if cuda.is_available():
    G = nn.DataParallel(G.cuda())

learner = RHingeGanLearner(G,None)
learner.load_generator("path-to-gen-model")

if __name__ == "__main__":
    "Define an instance of the normal distribution"
    dist = Normal(0,1)

    #Get a sample latent vector from the distribution
    latent_vector = dist.sample((1,128))

    #Define the class of the image you want to generate
    label = torch.LongTensor(1).fill_(5)

    #Run inference
    image = learner.predict([latent_vector,label])

    #Save generated image
    save_image(image, "image.jpg")
    print("SAVED")


