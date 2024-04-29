import torch
import numpy as np
import sys

use_cuda = torch.cuda.is_available()

model_type_gt = "gt_layout"
model_type_scratch = "scratch"
model_type_gt_rl = "gt+rl"
image_sets = ['train.large', 'train.med', 'train.small', 'train.tiny']
vocab_question_file = '/vocabulary_shape.txt'
vocab_layout_file = '/vocabulary_layout.txt'
training_text_files = '/%s.query_str.txt'
training_image_files = '/%s.input.npy'
training_label_files = '/%s.output'
training_gt_layout_file = '/%s.query_layout_symbols.json'
image_mean_file = '/image_mean.npy'
image_feat_file = '/image_feat.npy'


class HyperParameter:
    def __init__(self, model_type):
        if model_type == model_type_gt:
            # Module parameters
            self.H_im = 30
            self.W_im = 30
            self.D_feat = 64
            self.embed_dim_que = 300
            self.embed_dim_nmn = 300
            self.lstm_dim = 256
            self.num_layers = 2
            self.encoder_dropout = 0
            self.decoder_dropout = 0
            self.decoder_sampling = True
            self.T_encoder = 15
            self.T_decoder = 10
            self.batch_size = 128
            self.prune_filter_module = True

            # Training parameters
            self.weight_decay = 5e-4
            self.baseline_decay = 0.99
            self.max_grad_l2_norm = 10
            self.max_iter = 40000
            self.snapshot_interval = 10000
            self.lambda_entropy = 0
            self.learning_rate = 0.001
        elif model_type == model_type_scratch:
            # Module parameters
            self.H_feat = 10
            self.W_feat = 15
            self.D_feat = 512
            self.embed_dim_que = 300
            self.embed_dim_nmn = 300
            self.lstm_dim = 512
            self.num_layers = 2
            self.encoder_dropout = False
            self.decoder_dropout = False
            self.decoder_sampling = True
            self.T_encoder = 45
            self.T_decoder = 10
            self.batch_size = 64
            self.prune_filter_module = True

            # Training parameters
            self.invalid_expr_loss = np.log(28)  # loss value when the layout is invalid
            self.lambda_entropy = 0.01
            self.weight_decay = 0
            self.baseline_decay = 0.99
            self.max_grad_l2_norm = 10
            self.max_iter = 120000
            self.snapshot_interval = 10000
            self.learning_rate = 0.001
        else:
            sys.exit("unknown model type %s" % model_type)
