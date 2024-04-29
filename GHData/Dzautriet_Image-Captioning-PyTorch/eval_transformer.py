# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 19:07:35 2020

@author: Zhe Cao
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchtext.data.metrics import bleu_score
import os
import gc
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models
from load_data import Flickr30kDataset
from transformer import Encoder, Decoder
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

data_path = 'flickr30k_images/flickr30k_images'
csv_path = 'flickr30k_images/results.csv'
save_path = "."
use_pretrained_embed = False

epochs = 100
batch_size = 32
enc_lr = 1e-4
dec_lr = 1e-4
patience = 10
enc_save_path, dec_save_path = os.path.join(save_path, 'best_enc_trans'), os.path.join(save_path, 'best_dec_trans')
best_acc = 0
best_epoch = 0
channels = 2048
emb_size = 300
nHead = 10
hidden_size = 300
nLayers = 3
dropout_dec = 0.2
dropout_pos = 0.1

def token_sentence(decoder_out, itos):
    tokens = decoder_out
    tokens = tokens.transpose(1, 0)
    tokens = tokens.cpu().numpy()
    results = []
    for instance in tokens:
        result = ' '.join([itos[x] for x in instance])
        results.append(''.join(result.partition('<eos>')[0])) # Cut before '<eos>'
    return results

if __name__ == "__main__":
    # Instantiate data set
    transforms_train = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    data_set = Flickr30kDataset(data_path, csv_path, transforms_train, 3)
    reference_corpus = data_set.get_reference_corpus()
    vocab_size = len(data_set.field.vocab.itos)
    
    # Split data set
    num_data = len(data_set)
    idx = list(range(num_data))
    train_set = Subset(data_set, idx[:-2000])
    vali_set = Subset(data_set, idx[-2000:-1000])
    test_set = Subset(data_set, idx[-1000:])

    # Data loader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    vali_loader = DataLoader(vali_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)
    
    # Instantiate models
    resnet = models.resnet50(pretrained=True)
    encoder = Encoder(resnet, channels, emb_size)
    encoder.freeze_all()
    decoder = Decoder(vocab_size, emb_size, nHead, hidden_size, nLayers, dropout_dec, dropout_pos, use_pretrained_embed, embedding_matrix=None)
    decoder, encoder = decoder.cuda(), encoder.cuda()
    del resnet
    gc.collect()
    torch.cuda.empty_cache()
    
    # Load trained models
    encoder.load_state_dict(torch.load(enc_save_path))
    decoder.load_state_dict(torch.load(dec_save_path))
    encoder.eval()
    decoder.eval()
    
    itos = data_set.field.vocab.itos
    pred_len = data_set.seq_len - 1
    result_collection = []
    
    # Decode with greedy
    with torch.no_grad():
        for batch_index, (inputs, captions, caplens) in enumerate(test_loader):
            inputs, captions = inputs.cuda(), captions.cuda()
            enc_out = encoder(inputs)
            captions_input = captions[:, :-1]
            captions_target = captions[:, 1:]
            output = decoder.pred(enc_out, pred_len)
            result_caption = token_sentence(output, itos)
            result_collection.extend(result_caption)
            
    # Visualise an example
    i = 15
    plt.imshow(Image.open(os.path.join(data_path, data_set.data_files[-1000+i])))
    plt.axis('off')
    plt.show()
    print("Ground truth:", data_set.caption_list[-1000+i])
    print("Prediction-greedy:", result_collection[i])
    
    # Bleu scores w.r.t. all candidates
    uni_bleu = bleu_score([x.split(' ') for x in result_collection], reference_corpus[-1000:], max_n=1, weights=[1])
    bi_bleu = bleu_score([x.split(' ') for x in result_collection], reference_corpus[-1000:], max_n=2, weights=[1/2]*2)
    tri_bleu = bleu_score([x.split(' ') for x in result_collection], reference_corpus[-1000:], max_n=3, weights=[1/3]*3)
    qua_bleu = bleu_score([x.split(' ') for x in result_collection], reference_corpus[-1000:], max_n=4, weights=[1/4]*4)
    print("BLEU-1:, {:.4f}, BLEU-2:, {:.4f}, BLEU-3:, {:.4f}, BLEU-4:, {:.4f}".format(uni_bleu, bi_bleu, tri_bleu, qua_bleu))
    
    