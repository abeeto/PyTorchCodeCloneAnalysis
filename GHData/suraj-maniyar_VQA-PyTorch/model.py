import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import os


class LanguageModel(nn.Module):
    def __init__(self, config):
        super(LanguageModel, self).__init__()
        self.config = config
        self.hidden_dim = self.config['num_hidden_units']
        self.num_layers = self.config['num_layers']

        self.lstm = nn.LSTM(input_size=self.config['embedding_size'],
                                                hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=self.config['dropout'])

        self.fc = nn.Linear(4*self.hidden_dim, config['question_feature'])


    def forward(self, x):
                
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim))
            c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim))
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
                
        out1, out2, out3, out4 = hn[0], cn[0], hn[1], cn[1]        
        out = torch.cat((out1, out2, out3, out4), dim=1)

        out = self.fc(out)
        out = torch.tanh(out)

        return out



class VQA_FeatureModel(nn.Module):
    def __init__(self, config):
        super(VQA_FeatureModel, self).__init__()
        self.conig = config

        self.question_model = LanguageModel(config)

        self.bn1 = nn.BatchNorm1d(config['image_feature'])
        self.bn2 = nn.BatchNorm1d(config['question_feature']) 
        
        self.dropout = nn.Dropout(config['dropout']) 
        self.fc_image = nn.Linear(4096, config['image_feature'])    # VGG-16 feature vector has dimensions: 4096

        self.fc_combined_1 = nn.Linear(1024, 1024)
        self.dropout_1 = nn.Dropout(config['dropout'])

        self.fc_combined_2 = nn.Linear(1024, 1024)
        self.dropout_2 = nn.Dropout(config['dropout'])

        self.fc_combined_final = nn.Linear( 1024, config['vocab_size'])
         

    def forward(self, x1, x2):
        
        
        image_feature = self.fc_image(x1)
        question_feature = self.question_model(x2)

        image_feature = torch.tanh(image_feature)
        
        image_feature = self.bn1(image_feature)
        question_feature = self.bn2(question_feature) 

        concat = torch.mul(image_feature, question_feature)
        concat = self.dropout(concat)        

        out = self.fc_combined_1(concat)
        out = torch.tanh(out)
        out = self.dropout_1(out)

        out = self.fc_combined_2(out)
        out = torch.tanh(out)
        out = self.dropout_2(out)  
     
        out = self.fc_combined_final(out)

        return out




class VGG16(nn.Module):

     def __init__(self):

         super().__init__()

         self.vgg16 = models.vgg16(pretrained=True)
         self.layer1 = nn.Sequential(*list(self.vgg16.features.children()))
         self.layer3 = nn.Sequential(*list(self.vgg16.classifier.children())[0:4])

     def forward(self, x):
         out = self.layer1(x)
         out = out.view(out.size(0),-1)
         out = self.layer3(out)

         return out
