import multiprocessing.dummy as mp

import torch
import math
import torch.multiprocessing
import torch.nn.functional as F
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import nn, Tensor
from torchinfo import summary
from torchvision import models

torch.multiprocessing.set_sharing_strategy('file_system')


class TransformerModel(nn.Module):

    def __init__(self, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.5, num_classes: int = 2):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, input: Tensor):
        src = self.pos_encoder(input)
        output = self.transformer_encoder(src)
        output = output.reshape(-1, self.d_model)
        predictions = self.fc(output)
        predictions_soft = nn.functional.softmax(predictions, dim=1)

        return predictions_soft


######################################################################
# ``PositionalEncoding`` module injects some information about the
# relative or absolute position of the tokens in the sequence. The
# positional encodings have the same dimension as the embeddings so that
# the two can be summed. Here, we use ``sine`` and ``cosine`` functions of
# different frequencies.
#

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class FFNet(nn.Module):

    def __init__(self, num_dims, num_inputs=2):
        super().__init__()
        self.fc1 = nn.Linear(num_dims * num_inputs, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x.reshape(-1)


class EmbeddingNet(nn.Module):

    def __init__(self, num_dims):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features

        self.model.fc = nn.Linear(num_ftrs, num_dims)
        self.model.cuda()

    def forward(self, x):
        return self.model(x)

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetEfficientNetB0(nn.Module):

    def __init__(self, num_dims):
        super().__init__()
        self.model = models.efficientnet_b0(pretrained=True)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, num_dims)
        print(summary(self.model, input_size=(32, 3, 100, 100)))

    def forward(self, x):
        return self.model(x)

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetEfficientNetB2(nn.Module):

    def __init__(self, num_dims):
        super().__init__()
        self.model = models.efficientnet_b2(pretrained=True)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, num_dims)
        print(summary(self.model, input_size=(32, 3, 100, 100)))

    def forward(self, x):
        return self.model(x)

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetDenseNet(nn.Module):

    def __init__(self, num_dims):
        super().__init__()
        self.model = models.densenet121(pretrained=True)
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, num_dims)
        print(summary(self.model, input_size=(32, 3, 100, 100)))

    def forward(self, x):
        return self.model(x)

    def get_embedding(self, x):
        return self.forward(x)


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Resnet(nn.Module):

    def __init__(self, extract=False):
        super().__init__()
        self.extract = extract
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        # self.model.fc = nn.Linear(num_ftrs, num_dims)
        # num_ftrs = self.model.fc.in_features
        # if extract:
        #    self.out_layer = Identity()
        # else:
        #    self.out_layer = nn.Linear(num_dims, 3)
        if self.extract:
            self.model.fc = nn.Identity()
        else:
            self.model.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        output = self.model(x)
        if self.extract:
            return output
        else:
            predictions = nn.functional.softmax(output)
            return predictions

    def get_embedding(self, x):
        return self.forward(x)


class SiameseNetV2(nn.Module):

    def __init__(self, num_dims: int = 256):
        super().__init__()

        self.embedding_net = EmbeddingNet(num_dims)
        self.pool = mp.Pool(processes=2)
        self.fc = nn.Linear(num_dims, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1, x2 = x
        output1, output2 = self.pool.map(self.get_embedding, [x1, x2])
        l1_distance = torch.abs(output1 - output2)
        out = self.fc(l1_distance)
        y_prob = self.sigmoid(out)
        return y_prob.reshape(-1)

    def get_embedding(self, x):
        return self.embedding_net(x)


class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bidirectional=True):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional,
                            dropout=0.5)
        hidden_size_fc = hidden_size * 2 if self.bidirectional else hidden_size
        self.fc = nn.Linear(hidden_size_fc, num_classes)  # 2 for bidirection

    def forward(self, x):
        vector_size = self.num_layers * 2 if self.bidirectional else self.num_layers
        # Set initial states
        h0 = torch.zeros(vector_size, x.size(0), self.hidden_size).cuda()  # 2 for bidirection
        c0 = torch.zeros(vector_size, x.size(0), self.hidden_size).cuda()

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step
        out = self.fc(out)

        out = out.reshape(-1, self.num_classes)
        predictions = nn.functional.softmax(out)
        return predictions

class EyeLRCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.dims = 256
        self.sequence_len = 64
        self.siamese = SiameseNetV2(self.dims)
        self.lstm = BiRNN(self.dims, 256, 2, 2)

    def forward(self, x):
        features = self.siamese.get_embedding(x)
        features = features.reshape(-1, self.sequence_len, self.dims)
        outputs = self.lstm(features)
        return outputs



