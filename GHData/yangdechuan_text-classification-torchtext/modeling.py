import torch
import torch.nn as nn
import torch.nn.functional as F


class KimCNN(nn.Module):
    def __init__(self, word_embedding, num_classes):
        super(KimCNN, self).__init__()
        self.word_embedding = nn.Embedding.from_pretrained(word_embedding,
                                                           freeze=False)
        embedding_dim = word_embedding.shape[1]
        self.conv13 = nn.Conv2d(1, 100, kernel_size=(3, embedding_dim))
        self.conv14 = nn.Conv2d(1, 100, kernel_size=(4, embedding_dim))
        self.conv15 = nn.Conv2d(1, 100, kernel_size=(5, embedding_dim))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(300, num_classes)

    def _conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(dim=3)  # (B, 100, L - i + 1)
        x = F.max_pool1d(x, x.size(2)).squeeze(dim=2)  # (B, 100)
        return x

    def forward(self, sentence):
        """
        Arguments:
            sentence: Tensor of shape (batch_size, max_len)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        embs = self.word_embedding(sentence)  # (B, L, H)
        embs = embs.unsqueeze(dim=1)  # (B, 1, L, H)

        cnn_out1 = self._conv_and_pool(embs, self.conv13)  # (B, 100)
        cnn_out2 = self._conv_and_pool(embs, self.conv14)  # (B, 100)
        cnn_out3 = self._conv_and_pool(embs, self.conv15)  # (B, 100)
        cnn_out = torch.cat((cnn_out1, cnn_out2, cnn_out3), dim=1)  # (B, 300)
        cnn_out = self.dropout(cnn_out)  # (B, 300)
        logits = self.fc(cnn_out)  # (B, C)

        return logits
