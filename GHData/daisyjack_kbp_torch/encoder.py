import torch
import torch.nn as nn
import codecs
from configurations import config, to_np
from batch_getter import BatchGetter, get_source_mask
from torch.autograd import Variable
import torch.nn.init
import torch.nn.functional as F

class LoadEmbedding(object):
    def __init__(self, emb_file):
        with codecs.open(emb_file, mode='rb', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    if i == 0:
                        parts = line.split(' ')
                        self.voc_size = int(parts[0]) + 8
                        self.emb_size = int(parts[1])
                        self.embedding_tensor = torch.FloatTensor(self.voc_size, self.emb_size)
                    else:
                        parts = line.split(' ')
                        for j, part in enumerate(parts[1:]):
                            self.embedding_tensor[i+2, j] = float(part)

    def get_embedding_tensor(self):
        return self.embedding_tensor

    def get_voc_size(self):
        return self.voc_size

    def get_emb_size(self):
        return self.emb_size





class CNNEncoder(nn.Module):
    def __init__(self, loaded_embedding, dropout_p=config['dropout']):
        super(CNNEncoder, self).__init__()
        self.loaded_embedding = loaded_embedding
        self.dropout = nn.Dropout(dropout_p)

        self.out_dim = 0
        self.input_dim = 0
        self.embedding0 = nn.Embedding(self.loaded_embedding.get_voc_size(), self.loaded_embedding.get_emb_size())
        self.embedding0.weight.data.copy_(self.loaded_embedding.get_embedding_tensor())
        self.input_dim += 1
        self.out_dim += self.loaded_embedding.get_emb_size()
        self.embeddings = 1
        if len(config['EmbNames']) > 1:
            for (idx, embName) in enumerate(config['EmbNames'][1:]):
                vocsize =config['Vocabs'][idx+1].getVocSize()
                embdim = config['EmbSizes'][idx+1]
                emb = nn.Embedding(vocsize, embdim)
                torch.nn.init.uniform(emb.weight, -config['weight_scale'], config['weight_scale'])
                self.add_module('embedding'+str(idx+1), emb)
                self.out_dim += embdim
                self.input_dim += 1
                self.embeddings += 1
        self._use_gaz = config['use_gaz']
        self.gemb = None
        if self._use_gaz:
            gazsize = len(config['Gazetteers'])
            gazdim = config['gaz_emb_dim']
            self.gazsize = gazsize
            self.gemb = nn.Linear(gazsize, gazdim, bias=False)
            torch.nn.init.uniform(self.gemb.weight, -config['weight_scale'], config['weight_scale'])
            self.input_dim += gazsize
            self.out_dim += gazdim

        self._use_char_conv = config['use_char_conv']
        self.char_emb = None
        self.char_conv = None
        if self._use_char_conv:
            vocChar = config['CharVoc']
            maxCharLen = config['max_char']
            self.char_len = maxCharLen
            char_emb_size = config['char_emb_dim']
            self.char_emb = nn.Embedding(vocChar.getVocSize(), char_emb_size)
            torch.nn.init.uniform(self.char_emb.weight, -config['weight_scale'], config['weight_scale'])
            self.input_dim += maxCharLen * 2
            self.char_conv = nn.Conv1d(in_channels=char_emb_size, out_channels=char_emb_size, kernel_size=3, stride=1, padding=1)
            torch.nn.init.uniform(self.char_conv.weight, -config['weight_scale'], config['weight_scale'])
            torch.nn.init.constant(self.char_conv.bias, 0)
            self.out_dim += char_emb_size
            self.conv_active = nn.ReLU()



        self.conv0 = nn.Conv1d(in_channels=self.out_dim, out_channels=config['encoder_filter_num'],
                               kernel_size=config['filter_size'], stride=1, padding=1)
        torch.nn.init.uniform(self.conv0.weight, -config['weight_scale'], config['weight_scale'])
        torch.nn.init.constant(self.conv0.bias, 0)

        for i in range(1, 3):
            this_cnn = nn.Conv1d(in_channels=config['encoder_filter_num'], out_channels=config['encoder_filter_num'],
                               kernel_size=config['filter_size'], stride=1, padding=1)
            torch.nn.init.uniform(this_cnn.weight, -config['weight_scale'], config['weight_scale'])
            torch.nn.init.constant(this_cnn.bias, 0)
            self.add_module('conv'+str(i), this_cnn)
        self.relu = nn.ReLU()

        self.encoder_gru = nn.GRU(input_size=config['encoder_filter_num'], hidden_size=config['hidden_size'] / 2, num_layers=1, bidirectional=True)
        for name, param in self.encoder_gru.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant(param, 0)
            elif 'weight' in name:
                torch.nn.init.orthogonal(param)

    # input: (batch, seq_length, 60) h_0: (num_layers * num_directions, batch, hidden_size)
    def forward(self, step, input, h_0, seq_length):
        self.encoder_gru.flatten_parameters()

        input_size = input.size()

        outs = []
        for idx in range(self.embeddings):
            # outs.append(emb.apply(input_[:,:,idx:(idx+1)]))
            emb = getattr(self, 'embedding'+str(idx))
            word_slice = input[:, :, idx]
            word_emb = emb(word_slice)
            outs.append(word_emb)

        curr_end = self.embeddings
        if self._use_gaz:
            gazStart = curr_end
            gazEnd = gazStart + self.gazsize
            if config['USE_CUDA']:
                outs.append(self.gemb(input[:, :, gazStart:gazEnd].type(torch.cuda.FloatTensor).view(-1, self.gazsize)).view(input_size[0], input_size[1], -1).contiguous())
            else:
                outs.append(self.gemb(input[:, :, gazStart:gazEnd].type(torch.FloatTensor).view(-1, self.gazsize)).view(input_size[0], input_size[1], -1).contiguous())
            curr_end = gazEnd

        if self._use_char_conv:
            chars = input[:, :, curr_end:curr_end + self.char_len].contiguous()
            chars_mask = input[:, :, (curr_end + self.char_len):(curr_end + 2 * self.char_len)]
            if config['USE_CUDA']:
                chars_mask = chars_mask.type(torch.cuda.FloatTensor)
            else:
                chars_mask = chars_mask.type(torch.FloatTensor)
            chars_size = chars.size()
            char_view = chars.view(-1, self.char_len)
            char_emb_out = self.char_emb(char_view)


            # char_shape = char_emb_out.shape
            # char_emb_out = char_emb_out.reshape((char_shape[0] * char_shape[1], char_shape[2], 1, char_shape[3]))
            # char_conv_out = self.char_conv.apply(char_emb_out)
            # char_conv_out = self.conv_active.apply(char_conv_out)
            # char_conv_out = char_conv_out.reshape(char_shape)
            # char_conv_out = char_conv_out * chars_mask.dimshuffle(0, 1, 2, 'x')
            # char_conv_out = tensor.max(char_conv_out, axis=2)

            char_emb_out = char_emb_out.transpose(1, 2)
            char_conv_out = self.char_conv(char_emb_out)
            char_conv_out = self.conv_active(char_conv_out)
            char_conv_out = char_conv_out.transpose(1, 2)
            chars_mask = chars_mask.view(-1, self.char_len)
            char_conv_out = char_conv_out * chars_mask.unsqueeze(2).expand_as(char_conv_out)
            char_conv_out, _ = torch.max(char_conv_out, 1)
            char_conv_out = char_conv_out.view(chars_size[0], chars_size[1], -1)
            outs.append(char_conv_out)


        output = torch.cat(outs, dim=-1)
        mask = Variable(get_source_mask(input_size[0], self.out_dim, input_size[1], seq_length))
        if config['USE_CUDA']:
            mask = mask.cuda(config['cuda_num'])
        if config['use_multi']:
            mask = mask.cuda(input.get_device())
        mask = mask.transpose(0, 1)
        embedded = output * mask
        embedded = self.dropout(embedded)
        # logger.histo_summary('embedded', to_np(embedded), step)


        # embedded = self.embedding0(input)  # embedded: (batch, seq_length, emb_size)
        output = embedded.transpose(1, 2)  # embedded: (batch, emb_size, seq_length)
        for i in range(3):
            conv = getattr(self, 'conv'+str(i))
            output = self.relu(conv(output))  # output: (batch, encoder_filter_num, seq_length)
            output = self.dropout(output)

        output = output.transpose(1, 2)
        output = output.transpose(0, 1).contiguous()  # output: (seq_length, batch, encoder_filter_num)
        # return self.dropout(output)

        gru_output, h_n = self.encoder_gru(output, h_0)
        return self.dropout(gru_output)  # (seq_len, batch, hidden_size * num_directions)


class MultiCNNEncoder(CNNEncoder):
    def __init__(self, loaded_embedding, dropout_p=config['dropout']):
        super(MultiCNNEncoder, self).__init__(loaded_embedding, dropout_p)

    # input: (batch, seq_length) h_0: (batch, num_layers * num_directions, hidden_size)
    def forward(self, step, input, h_0, seq_length):
        step = step.data[0,0]
        h_0 = h_0.transpose(0, 1).contiguous()
        seq_length = seq_length.cpu().data.numpy().reshape(-1).tolist()
        # print 'step', step, 'input', input.size(), 'h0', h_0.size(), 'seq', len(seq_length)
        result = super(MultiCNNEncoder, self).forward(step, input, h_0, seq_length)  # (seq_len, batch, hidden_size * num_directions)
        return result.transpose(0, 1)










if __name__ == '__main__':
    emb = LoadEmbedding('res/emb.txt')
    print emb.get_embedding_tensor()[0:5, :]
    encoder = CNNEncoder(emb)

    batch_getter = BatchGetter('data/train')
    data = Variable(next(batch_getter)[0])
    if config['USE_CUDA']:
        encoder.cuda(config['cuda_num'])
        data = data.cuda(config['cuda_num'])
    print data
    print encoder(data)
    pass


