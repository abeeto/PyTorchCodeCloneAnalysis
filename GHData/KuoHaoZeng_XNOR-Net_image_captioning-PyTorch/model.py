import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

backbone_zoo = {'resnet152': models.resnet152,\
                'resnet101': models.resnet101,\
                'resnet50': models.resnet50,\
                'resnet34': models.resnet34,\
                'resnet18': models.resnet18,\
                'densenet161': models.densenet161,\
                'densenet201': models.densenet201,\
                'densenet169': models.densenet169,\
                'densenet121': models.densenet121,\
                'inception_v3': models.inception_v3,\
                }

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, backbone = 'resnet18'):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        backbone = backbone_zoo[backbone](pretrained=True)
        modules = list(backbone.children())[:-1]      # delete the last fc layer.
        self.backbone = nn.Sequential(*modules)
        self.linear = nn.Linear(backbone.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        #with torch.no_grad():
        features = self.backbone(images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(0), embeddings.transpose(0,1)), 0)
        packed = pack_padded_sequence(embeddings, lengths) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        inputs = features.unsqueeze(0)
        hiddens, states = self.lstm(inputs, states)          # hiddens: (1, batch_size, hidden_size)
        outputs = self.linear(hiddens.squeeze(0))            # outputs:  (batch_size, vocab_size)
        _, predicted = outputs.max(1)                        # predicted: (batch_size)
        sampled_ids = predicted.unsqueeze(0)
        inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
        inputs = inputs.unsqueeze(0)                         # inputs: (1, batch_size, embed_size)
        for i in range(self.max_seg_length-1):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (1, batch_size, hidden_size)
            outputs = self.linear(hiddens.squeeze(0))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids = torch.cat((sampled_ids, predicted.unsqueeze(0)), 1)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(0)                         # inputs: (batch_size, 1, embed_size)
        return sampled_ids

class DecoderRNN_onnx(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN_onnx, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
    
    def forward(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        inputs = features.unsqueeze(0)
        hiddens, states = self.lstm(inputs, states)          # hiddens: (1, batch_size, hidden_size)
        outputs = self.linear(hiddens.squeeze(0))            # outputs:  (batch_size, vocab_size)
        _, predicted = outputs.max(1)                        # predicted: (batch_size)
        sampled_ids = predicted.unsqueeze(0)
        inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
        inputs = inputs.unsqueeze(0)                         # inputs: (1, batch_size, embed_size)
        for i in range(self.max_seg_length-1):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (1, batch_size, hidden_size)
            outputs = self.linear(hiddens.squeeze(0))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids = torch.cat((sampled_ids, predicted.unsqueeze(0)), 1)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(0)                         # inputs: (batch_size, 1, embed_size)
        return sampled_ids
