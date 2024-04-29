import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, model_info, vocab_size = 100000, embedding_dim = 300, use_embeddings = True, embeddings = []):
        super(Net, self).__init__()
        self.embeddings = self.create_embedding_layer(vocab_size, embedding_dim, embeddings, use_embeddings)
        self.conv_layers = self.create_conv_layers(model_info['conv_info'], embedding_dim)
        self.linear_layers = self.create_linear_layers(model_info['linear_info'])
        self.output_layer = nn.Linear(model_info['output_info']['in'], model_info['output_info']['out'])
        self.use_dropout = model_info["dropout"]["use"] 
        self.hidden_size = model_info["linear_info"][0]["out"]
        if self.use_dropout:
            self.dropout_layer = nn.Dropout(p = model_info["dropout"]["rate"])

    def forward(self, x):
        out = self.embeddings(x)
        out = torch.reshape(out, (out.size(0),1, -1))

        for layer in (self.conv_layers + self.linear_layers):
            out = layer(out)

        pool_shape = out.size(2)
       
        if self.use_dropout:
            out = self.dropout_layer(out)

        out = out.reshape(out.size(0), -1)

        out = F.relu(nn.Linear(pool_shape*self.last_conv_out, self.hidden_size)(out))
        out = F.softmax(self.output_layer(out))

        return out

    def create_conv_layers(self, conv_info, embedding_dim):
        layers = []

        for info in conv_info:
            self.last_conv_out = info['out']
            layers.append(nn.Sequential(
                nn.Conv1d(info['in'], info['out'], kernel_size = info['size']*embedding_dim, stride = embedding_dim, padding= (info['size'] - 1)*embedding_dim),
                nn.ReLU(),
                nn.MaxPool1d(info['pool']) 
                ) 
            )

        return layers

    def create_linear_layers(self, linear_info):
        layers = []

        for info in linear_info[1:]:
            layers.append(nn.Sequential(
                nn.Linear(info['in'], info['out']),
                nn.ReLU(),
                )
            )

        return layers

    def create_embedding_layer(self, vocab_size, embedding_dim, embeddings, use_embeddings):
        layer = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embedding_dim)
        
        if(use_embeddings):
            layer.load_state_dict({'weight': embeddings})
            layer.weight.requires_grad = False
       
        return layer

