from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

import pytorch_lightning as pl


class LitSentiment(pl.LightningModule):
    def __init__(self, vocab_size, embed_dim, num_class, learning_rate=4.0, lr_gamma=0.8):
        super().__init__()
        self.save_hyperparameters()
        self.embedding = nn.EmbeddingBag(self.hparams.vocab_size, self.hparams.embed_dim, sparse=True)
        self.fc = nn.Linear(self.hparams.embed_dim, self.hparams.num_class)
        self.init_weights()
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc = pl.metrics.Accuracy()

    def metric(self, preds, labels, step='val'):
        return {f'{step}_acc': self.acc(preds, labels)}

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets, cls=None):
        r"""
        Arguments:
            text: 1-D tensor representing a bag of text tensors
            offsets: a list of offsets to delimit the 1-D text tensor
                into the individual sequences.

        """
        logits = self.fc(self.embedding(text, offsets))
        if cls is None:
            return (logits,)
        return (self.loss_fn(logits, cls), logits)

    def training_step(self, batch, batch_idx):
        text, offsets, cls = batch
        loss, _ = self(text, offsets, cls)
        return loss

    def validation_step(self, batch, batch_idx):
        text, offsets, cls = batch
        loss, logits = self(text, offsets, cls)
        self.log('val_loss', loss, prog_bar=True)
        self.log_dict(self.metric(logits, cls), prog_bar=True)

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = StepLR(optimizer, 1, gamma=self.hparams.lr_gamma)
        return [optimizer], [scheduler]
