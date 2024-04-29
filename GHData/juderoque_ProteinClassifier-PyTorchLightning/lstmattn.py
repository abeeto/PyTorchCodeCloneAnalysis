import torch
import torch.nn as nn
import pytorch_lightning as pl
from utils.layers import LinearClassifier, MaskedLanguageModel, EncoderLayer, PositionalEncoding
import numpy as np


class LSTMAttn(pl.LightningModule):
    def __init__(self, embedding_dim,
                 hidden_dim,
                 seq_len,
                 vocab,
                 tagset_size,
                 lr=1e-4,
                 padding_idx=0,
                 d_k=64,
                 d_v=64,
                 n_lstm=3,
                 n_head=12,
                 n_attn=2,
                 dropout=0.1,
                 need_pos_enc=True,
                 mask_ratio=0.1,
                 cls_weight=0.5,
                 mlm_weight=0.5):
        super(LSTMAttn, self).__init__()

        self.lr = lr

        self.hidden_dim = hidden_dim
        self.d_k = d_k
        self.n_head = n_head
        self.n_attn = n_attn
        self.n_lstm = n_lstm
        self.need_pos_enc = need_pos_enc
        self.mask_ratio = mask_ratio
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.cls_weight = cls_weight
        self.mlm_weight = mlm_weight

        self.word_embeddings = nn.Embedding(self.vocab_size, embedding_dim, padding_idx=padding_idx)
        
        self.class_loss_function = nn.NLLLoss()
        self.lm_loss_function = nn.NLLLoss(ignore_index=0)

        if need_pos_enc:
            self.position_enc = PositionalEncoding(embedding_dim, n_position=seq_len)

        if n_lstm:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=n_lstm)
        else:
            # if we decided not to use a LSTM there, means there will be no "hidden dimension",
            # and simply we use embedding dimension for the further part.
            # we treat LSTM as transparent.
            # Divide 2 because the bi-LSTM actually output hidden_dim * 2
            self.hidden_dim = embedding_dim // 2
        if n_attn:
            self.hid2hid = nn.Linear(seq_len, 1)
        else:
            self.hid2hid = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)

        # self.slf_attn_encoder = EncoderLayer(n_head, self.hidden_dim, seq_len, d_k, d_v, dropout=dropout)
        self.slf_attn_encoder_blocks = nn.ModuleList([
            EncoderLayer(n_head, self.hidden_dim, seq_len, d_k, d_v, dropout=dropout)
            for _ in range(n_attn)])
        self.linear_classifier = LinearClassifier(self.hidden_dim, tagset_size)

        # because of the bi-LSTM the total hidden_dim should be *2
        # if not using bi-LSTM, the hidden_dim is first devided by 2
        self.masked_language_model = MaskedLanguageModel(
            self.hidden_dim * 2, self.vocab_size)
        self.dropout = nn.Dropout(p=dropout)
        self.padding_idx = padding_idx
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, token_mask=None):
        b_size, seq_len = x.size()

        # creating mask and map the mask to the input
        if token_mask is not None:
            mask_label = self.vocab["<MASK>"]
            x_input = x.masked_fill(token_mask == 0, mask_label)
        else:
            x_input = x

        # embedding + LSTM part
        enc_output = self.word_embeddings(x_input)
        if self.need_pos_enc:
            enc_output = self.position_enc(enc_output)
        # enc_output = self.dropout(enc_output)
        if self.n_lstm:
            # here "if" is for flexibility of taking LSTM or not.
            # enc_output, _ = self.lstm(enc_output.view(b_size, seq_len, -1))
            # u will want to send length only to the max size of current batch
            # to the LSTM model
            enc_output, _ = self.lstm(enc_output.view(b_size, seq_len, -1))

        enc_output = enc_output.transpose(1, 2)

        # take self-attention on the hidden states of the LSTM model
        # start attention layer

        ###### self attention version ######
        if self.n_attn:
            attn_out = enc_output
            for slf_attn_encoder in self.slf_attn_encoder_blocks:
                attn_out, slf_attn_res = slf_attn_encoder(attn_out)
        ####################################
        else:
            # default, take the last layer as output of LSTM
            attn_out = enc_output
        # linear transformation and classification layer

        tag_scores, mlm_scores = None, None
        if self.cls_weight:
            tag_scores = self.linear_classifier(attn_out)

        # masked language model part
        if self.mlm_weight:
            mlm_scores = self.masked_language_model(attn_out)

        return tag_scores, mlm_scores

    def training_step(self, batch, batch_idx):

        x, target, token_mask = batch
        tag_scores, mlm_scores = self(x, token_mask)
        # analyze the loss
        if self.cls_weight:
            cls_loss = self.class_loss_function(tag_scores, target)
        else:
            cls_loss = 0.0

        # we only compute cross entropy for the masked: mask==1 part
        # we mask all the other tokens to be 0 and since the loss function
        # will ignore all the place of 0, it's actually only considering
        # the content within the mask
        if self.mlm_weight:
            x_mask = x.masked_fill(token_mask == 1, 0)
            mlm_loss = self.lm_loss_function(mlm_scores, x_mask)
        else:
            mlm_loss = 0.0

        loss = (self.cls_weight * cls_loss + self.mlm_weight * mlm_loss) / (self.cls_weight + self.mlm_weight)
        log = {'loss': loss}
        self.logger.experiment.log(log)
        return log

    def validation_step(self, batch, batch_idx):

        x, target, token_mask = batch
        score_pred, mlm_pred = model(x)
        
        if self.cls_weight:
            cls_loss = self.class_loss_function(score_pred, target)
        else:
            cls_loss = 0.0

        if self.mlm_weight:
            x_mask = x.masked_fill(token_mask == 1, 0)
            mlm_loss = self.lm_loss_function(mlm_pred, x_mask)
        else:
            mlm_loss = 0.0
            
        loss = (self.cls_weight * cls_loss + self.mlm_weight * mlm_loss) / (self.cls_weight + self.mlm_weight)

        #if score_pred is not None:
            #cls_pred_fold = np.array(torch.max(score_pred, 1)[1].tolist())
            #cls_acc = sum(target == cls_pred_fold) / len(target)

        #if mlm_pred is not None:
            #mlm_pred_fold = torch.max(mlm_pred, 1)[1]
            # this is the compare result
            #mlm_comp_res = mlm_pred_fold == x
            # this is where we care, we will set these position to be 1
            #care_part = token_mask == 0
            #mlm_masked_res = mlm_comp_res & care_part

            #mlm_acc = np.sum(mlm_masked_res.tolist()) / np.sum(care_part.tolist())

        #return {'mlm_acc': torch.tensor(mlm_acc or 0, dtype=torch.float64), 'cls_acc': torch.tensor(cls_acc or 0, dtype=torch.float64), 'val_loss': loss}
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        #mlm_acc = torch.stack([x['mlm_acc'] for x in outputs]).mean()
        #cls_acc = torch.stack([x['cls_acc'] for x in outputs]).mean()
        
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        log = {'val_loss': val_loss} #, 'mlm_acc': mlm_acc, 'cls_acc': cls_acc}
        self.logger.experiment.log(log)
        return {'log': log}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == '__main__':
    from utils.dataset import PfamDataset
    import argparse
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
    from test_tube import Experiment

    #exp = Experiment(version='3airf7c8')

    checkpoint_callback = ModelCheckpoint(
        filepath='checkpoints/trial.ckpt',
        verbose=True,
        monitor='val_loss',
        save_top_k=5,
        mode='min'
    )

    wandb_logger = WandbLogger(name='Trial', project='lstm-lightning')

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    args = parser.parse_args()

    data_dir = 'data/random_split'
    sample_every = 1
    padding = 3000

    dataset = PfamDataset(data_dir, sample_every, padding)
    model = LSTMAttn(80, 200, padding, dataset.vocab, dataset.target_size, args.learning_rate)

    train_loader = dataset.get_train_loader(batch_size=args.batch_size)
    val_loader = dataset.get_val_loader(batch_size=args.batch_size)
    
    trainer = pl.Trainer(max_epochs=20, logger=wandb_logger, val_check_interval=0.10, val_percent_check=0.1, gpus=[0], checkpoint_callback=checkpoint_callback, resume_from_checkpoint='checkpoints/_ckpt_epoch_1.ckpt')#, fast_dev_run=True) 
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)


