import torch
from torch import nn
from torch.nn.utils.weight_norm import weight_norm
from torchvision import models

from config import hparams
from vocab import load_vocab


class CNN(nn.Module):
    """
    Encoder，使用CNN提取图像特征
    """
    def __init__(self):
        super(CNN, self).__init__()
        # 自动下载预训练模型，这里使用restnet101
        model_ft = models.resnet101(pretrained=True)
        # 删除后2层，只使用预训练模型的前面几层训练
        modules = list(model_ft.children())[:-2]
        self.cnn = nn.Sequential(*modules)
        self.froze()

    def forward(self, img):
        # img: batch_size * 3 * 224 * 224
        batch_size = img.size(0)
        # features: batch_size * 2048 * 7 * 7
        features = self.cnn(img)
        # fea_map: batch_size * 49 * 2048
        fea_maps = features.permute(0, 2, 3, 1).contiguous().view(batch_size, 49, -1)
        return fea_maps

    def froze(self):
        # 冻结参数，不参与训练
        for param in self.cnn.parameters():
            param.requires_grad = False

    def get_params(self):
        return list(self.cnn.parameters())

    def fine_tune(self):
        for layer in list(self.cnn.children())[6:]:
            for param in layer.parameters():
                param.requires_grad = True


class RNN(nn.Module):
    """
    Decoder，使用RNN生成自然语言描述
    """
    def __init__(self, fea_dim, embed_dim, hid_dim, max_sen_len, vocab_pkl):
        super(RNN, self).__init__()
        self.fea_dim = fea_dim
        self.embed_dim = embed_dim
        self.hid_dim = hid_dim
        self.max_sen_len = max_sen_len
        self.vocab = load_vocab(vocab_pkl)

        self.vocab_size = self.vocab.__len__()
        self.lstm_cell = nn.LSTMCell(self.embed_dim, self.hid_dim)
        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)  # num_embeddings, embedding_dim
        self.fc = weight_norm(nn.Linear(self.hid_dim, self.vocab_size))
        self.dropout = nn.Dropout(0.5)
        self.init_h = weight_norm(nn.Linear(self.fea_dim, self.hid_dim))
        self.init_c = weight_norm(nn.Linear(self.fea_dim, self.hid_dim))

        self.init_weight()

    def init_hc(self, fea_vec):
        """使用图像特征初始化LSTM状态，即将fea_vec用线形变换投射到h, c"""
        h = self.init_h(fea_vec)
        c = self.init_c(fea_vec)
        return h, c

    def init_weight(self):
        """初始化网络层的权重"""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, fea_maps, caption):
        # fea_maps: batch_size * 49 * 2048
        batch_size = fea_maps.size(0)
        # fea_vec: batch_size * 2048 卷积核内的数据取平均
        # embedding: batch_size * 512
        h, c = self.init_hc(fea_maps.mean(1))
        # 每个单词使用embedding向量表示
        embeddings = self.embed(caption)
        outputs = torch.zeros(batch_size, self.max_sen_len - 1, self.vocab_size).type_as(fea_maps)
        # 训练过程中送入LSTM单元的是ground truth
        for t in range(self.max_sen_len - 1):
            h, c = self.lstm_cell(embeddings[:, t, :], (h, c))
            out = self.fc(self.dropout(h))
            outputs[:, t, :] = out

        return outputs

    def sample(self, fea_maps):
        """
        多项式抽样生成句子

        :param fea_maps:
        :return:
        """
        batch_size = fea_maps.size(0)
        h, c = self.init_hc(fea_maps.mean(1))
        # <start>
        input_id = torch.ones(batch_size).type_as(fea_maps).long()
        embedding = self.embed(input_id)
        # 二维数组存储生成的单词id：batch_size * max_sen_len
        output_ids = torch.zeros(batch_size, self.max_sen_len - 1).type_as(fea_maps).long()
        p = torch.zeros(batch_size, self.max_sen_len - 1).type_as(fea_maps)

        for t in range(self.max_sen_len - 1):
            # 验证过程送入LSTM生成词是预测的词
            h, c = self.lstm_cell(embedding, (h, c))
            fc = self.fc(self.dropout(h))
            predicted = torch.softmax(fc, 1)
            next_token = predicted.max(1)[1]  # (batch_size)

            for i in range(batch_size):
                p[i, t] = predicted[i, next_token[i]]
            output_ids[:, t] = next_token
            embedding = self.embed(next_token)
        sens, sen_lens = self.ids2sen(output_ids, p)
        return sens, sen_lens

    def ids2sen(self, word_ids, word_pros):
        """
        把ids转换成生成的句子，同时计算生成该句话的概率

        :param word_pros: 单词概率的矩阵 batch_size * (max_sen_len-1)
        :param word_ids: 单词id的矩阵 batch_size * (max_sen_len-1)
        """
        sens = []
        lens = []
        sen_pros = torch.zeros(word_pros.size(0)).type_as(word_pros)

        for i, each_sen_ids in enumerate(word_ids):
            each_sen_ids = map(int, each_sen_ids)
            p = word_pros[i]
            sentence = []
            p_j = torch.ones(1).type_as(word_pros)
            for j, word_id in enumerate(each_sen_ids):
                word = self.vocab.idx2word[word_id]
                if word == '<start>':
                    continue
                # <end>的概率先不算(CIDEr1)，现在计算<end>的概率,算了之后效果有下降
                p_j *= p[j - 1]  # ids第一个存的是<start>
                if word == '<end>':
                    break

                sentence.append(word)
            length = len(sentence) + 2 if len(sentence) <= 18 else 20
            lens.append(length)
            sens.append(' '.join(sentence))
            sen_pros[i] = p_j
            # sen_pros.cpu().tolist()
        return sens, lens


if __name__ == '__main__':
    rnn = RNN(2048, 512, 512, 20, hparams.vocab_pkl)
    print(rnn)
    print(rnn.parameters())
