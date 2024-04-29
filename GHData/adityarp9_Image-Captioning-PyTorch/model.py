"""
Model architecture
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as vis
from keras_preprocessing.sequence import pad_sequences
#from torchsummary import summary
from collections import OrderedDict as OD
import subprocess

import dataset as ds


class CapNet(nn.Module):

    def __init__(self, vocab_size, max_len_cap):
        super(CapNet, self).__init__()
        self.max_len_cap = max_len_cap
        embed_dim = 256
        hidden_size = 512
        self.embd = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size)
        self.fcn1 = nn.Linear(4096, hidden_size)
        self.fcn2 = nn.Linear(hidden_size, 256)
        self.fcn3 = nn.Linear(256, vocab_size)

    def forward(self, x_img, x_cap):
        x_cap = self.embd(x_cap)
        self.lstm.flatten_parameters()
        x_cap, _ = self.lstm(x_cap)
        x_img = self.fcn1(x_img)
        x_img = F.relu(x_img)
        latent = torch.add(x_cap[-1], x_img)
        x_vec = self.fcn2(latent)
        x_vec = F.relu(x_vec)
        x_vec = self.fcn3(x_vec)
        return x_vec

    def check_updation(self, grad=True):
        for name, p in self.named_parameters():
            print("-------------------------------------------------")
            print("param:")
            print(name, p, sep='\n')
            if grad:
                print("grad:")
                print(name, p.grad, sep='\n')
            print()

    def prediction(self, image, tokens, device):
        input_cap = "startseq"

        for i in range(self.max_len_cap):
            seq = tokens.texts_to_sequences([input_cap])[0]
            seq = torch.from_numpy(pad_sequences([seq], maxlen=self.max_len_cap)).long()

            out = F.softmax(self(image.to(device), seq.t().to(device)), dim=1)
            _, pred = torch.max(out, 1)
            new_word = ds.index2word(tokens.word_index)[pred.item()]
            input_cap += " " + new_word
            if new_word == "endseq":
                break
        return input_cap


def get_features(images, download_wts=False, save=False, cuda=False):

    if download_wts:
        print("Downloading model weights")
        subprocess.run(["wget", "https://download.pytorch.org/models/vgg16-397923af.pth", "-P", "models/"])


    ## Load model parameters from path
    vgg_net = vis.models.vgg16()
    vgg_net.load_state_dict(torch.load('./models/vgg16-397923af.pth'))

    for p in vgg_net.parameters():
        p.requires_grad = False

    ## Net architecture
    print(vgg_net)
    # summary(vgg_net, input_size=(3, 224, 224))

    ## Remove the last classifier layer: Softmax
    print("Removing softmax layer of VGG16 ... ")
    vgg_net.classifier = vgg_net.classifier[:-1]
    print(vgg_net)
    # summary(vgg_net, input_size=(3, 224, 224))

    # print(images.keys())

    ## Get feature map for image tensor through VGG-16
    img_featrs = OD()
    vgg_net.eval()
    if cuda:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    vgg_net.to(device)
    print("Using " + device) 
    start = time.time()
    print("Gathering images' features from last layer of %s ... " % type(vgg_net).__name__)
    for i, jpg_name in enumerate(images.keys()):
        with torch.no_grad():
            print(i, jpg_name)
            img_featrs[jpg_name] = vgg_net((images[jpg_name].unsqueeze(0)).to(device))
    elapsed = time.time() - start
    print("\nTime to fprop {} images VGG-16 on CPU: {:.2f} \
            seconds".format(i, elapsed))
    if save:
        print("Saving extracted features ... ", end="")
        features_fname = 'features_' + type(vgg_net).__name__ + "_" + device + '.pkl'
        torch.save(img_featrs, features_fname)
        print("done.")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return img_featrs, features_fname
