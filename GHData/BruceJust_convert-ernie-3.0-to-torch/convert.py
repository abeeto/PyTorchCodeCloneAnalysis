#! -*- coding: utf-8 -*-

# Author: Bruce
# Create type_time: 2022/7/4
# Info: convert.py

import collections
import os
import json
import shutil
import paddle
import paddle.fluid.dygraph as D
import torch
from paddle import fluid

# downloading paddlepaddle model
# ERNIE3.0: https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-3.0
from transformers import BertTokenizer, BertForMaskedLM, BertModel

# paddle.device.set_device('cpu')

def build_params_map(attention_num=12):
    """
    build params map from paddle-paddle's ERNIE to transformer's BERT
    :return:
    """
    weight_map = collections.OrderedDict({
        'ernie.embeddings.word_embeddings.weight': "bert.embeddings.word_embeddings.weight",
        'ernie.embeddings.position_embeddings.weight': "bert.embeddings.position_embeddings.weight",
        'ernie.embeddings.token_type_embeddings.weight': "bert.embeddings.token_type_embeddings.weight",
        'ernie.embeddings.layer_norm.weight': 'bert.embeddings.LayerNorm.gamma',
        'ernie.embeddings.layer_norm.bias': 'bert.embeddings.LayerNorm.beta',
    })
    # add attention layers
    for i in range(attention_num):
        weight_map[f'ernie.encoder.layers.{i}.self_attn.q_proj.weight'] = f'bert.encoder.layer.{i}.attention.self.query.weight'
        weight_map[f'ernie.encoder.layers.{i}.self_attn.q_proj.bias'] = f'bert.encoder.layer.{i}.attention.self.query.bias'
        weight_map[f'ernie.encoder.layers.{i}.self_attn.k_proj.weight'] = f'bert.encoder.layer.{i}.attention.self.key.weight'
        weight_map[f'ernie.encoder.layers.{i}.self_attn.k_proj.bias'] = f'bert.encoder.layer.{i}.attention.self.key.bias'
        weight_map[f'ernie.encoder.layers.{i}.self_attn.v_proj.weight'] = f'bert.encoder.layer.{i}.attention.self.value.weight'
        weight_map[f'ernie.encoder.layers.{i}.self_attn.v_proj.bias'] = f'bert.encoder.layer.{i}.attention.self.value.bias'
        weight_map[f'ernie.encoder.layers.{i}.self_attn.out_proj.weight'] = f'bert.encoder.layer.{i}.attention.output.dense.weight'
        weight_map[f'ernie.encoder.layers.{i}.self_attn.out_proj.bias'] = f'bert.encoder.layer.{i}.attention.output.dense.bias'
        weight_map[f'ernie.encoder.layers.{i}.norm1.weight'] = f'bert.encoder.layer.{i}.attention.output.LayerNorm.gamma'
        weight_map[f'ernie.encoder.layers.{i}.norm1.bias'] = f'bert.encoder.layer.{i}.attention.output.LayerNorm.beta'
        weight_map[f'ernie.encoder.layers.{i}.linear1.weight'] = f'bert.encoder.layer.{i}.intermediate.dense.weight'
        weight_map[f'ernie.encoder.layers.{i}.linear1.bias'] = f'bert.encoder.layer.{i}.intermediate.dense.bias'
        weight_map[f'ernie.encoder.layers.{i}.linear2.weight'] = f'bert.encoder.layer.{i}.output.dense.weight'
        weight_map[f'ernie.encoder.layers.{i}.linear2.bias'] = f'bert.encoder.layer.{i}.output.dense.bias'
        weight_map[f'ernie.encoder.layers.{i}.norm2.weight'] = f'bert.encoder.layer.{i}.output.LayerNorm.gamma'
        weight_map[f'ernie.encoder.layers.{i}.norm2.bias'] = f'bert.encoder.layer.{i}.output.LayerNorm.beta'
    # add pooler
    weight_map.update(
        {
            'pooler.weight': 'bert.pooler.dense.weight',
            'pooler.bias': 'bert.pooler.dense.bias',
            'mlm.weight': 'cls.predictions.transform.dense.weight',
            'mlm.bias': 'cls.predictions.transform.dense.bias',
            'mlm_ln.weight': 'cls.predictions.transform.LayerNorm.gamma',
            'mlm_ln.bias': 'cls.predictions.transform.LayerNorm.beta',
            'mlm_bias': 'cls.predictions.bias'
        }
    )
    return weight_map


def extract_and_convert(input_dir, output_dir, model_file):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print('=' * 20 + 'save config file' + '=' * 20)
    config = json.load(open(os.path.join(input_dir, 'model_config.json'), 'rt', encoding='utf-8'))
    config = config['init_args'][0]
    config['layer_norm_eps'] = 1e-5
    if 'sent_type_vocab_size' in config:
        config['type_vocab_size'] = config['sent_type_vocab_size']
    config['intermediate_size'] = 4 * config['hidden_size']
    config['num_hidden_layers'] = 5
    json.dump(config, open(os.path.join(output_dir, 'config.json'), 'wt', encoding='utf-8'), indent=4)
    print('=' * 20 + 'save vocab file' + '=' * 20)
    with open(os.path.join(input_dir, 'tiny_vocab.txt'), 'rt', encoding='utf-8') as f:
        words = f.read().splitlines()
    words = [word.split('\t')[0] for word in words]
    with open(os.path.join(output_dir, 'tiny_vocab.txt'), 'wt', encoding='utf-8') as f:
        for word in words:
            f.write(word + "\n")
    print('=' * 20 + 'extract weights' + '=' * 20)
    state_dict = collections.OrderedDict()
    weight_map = build_params_map(attention_num=config['num_hidden_layers'])
    # with fluid.dygraph.guard():
    #     paddle_paddle_params, _ = D.load_dygraph(os.path.join(input_dir, 'saved_weights'))
    state_dict_ = paddle.load(os.path.join(input_dir, model_file), return_numpy=True)
    # for weight_name, weight_value in paddle_paddle_params.items():
    for weight_name, weight_value in state_dict_.items():
        if 'linear' in weight_name:
            weight_value = weight_value.transpose()
        if weight_name not in weight_map:
            print('=' * 20, '[SKIP]', weight_name, '=' * 20)
            continue
        state_dict[weight_map[weight_name]] = torch.FloatTensor(weight_value)
        print(weight_name, '->', weight_map[weight_name], weight_value.shape)
    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))


if __name__ == '__main__':
    input_dir = '/root/.paddlenlp/models/ernie-3.0-nano-zh/'
    model_file = 'ernie_3.0_nano_zh.pdparams'
    output_dir = './convert'
    extract_and_convert(input_dir, output_dir, model_file)
    tokenizer = BertTokenizer.from_pretrained('./convert')
    model = BertModel.from_pretrained('./convert5')
    input_ids = torch.tensor([tokenizer.encode("hello", add_special_tokens=True)])
    with torch.no_grad():
        pooled_output = model(input_ids)[1]
        print(pooled_output.numpy())