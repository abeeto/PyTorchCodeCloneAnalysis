import torch
from src.centerface import Centerface

replace_dict = {
    'first_block.0.conv':'base.feature_1.0',
    'first_block.1.conv':'base.feature_1.1.conv.0',
    'first_block.2':'base.feature_1.1.conv.1',
    'first_block.3':'base.feature_1.1.conv.2',

    'inverted_residual_blocks.0.0.conv.0':'base.feature_1.2.conv.0.0',
    'inverted_residual_blocks.0.0.conv.1':'base.feature_1.2.conv.0.1',
    'inverted_residual_blocks.0.0.conv.3':'base.feature_1.2.conv.1.0',
    'inverted_residual_blocks.0.0.conv.4':'base.feature_1.2.conv.1.1',
    'inverted_residual_blocks.0.0.conv.6':'base.feature_1.2.conv.2',
    'inverted_residual_blocks.0.0.conv.7':'base.feature_1.2.conv.3',

    'inverted_residual_blocks.0.1.conv.0':'base.feature_1.3.conv.0.0',
    'inverted_residual_blocks.0.1.conv.1':'base.feature_1.3.conv.0.1',
    'inverted_residual_blocks.0.1.conv.3':'base.feature_1.3.conv.1.0',
    'inverted_residual_blocks.0.1.conv.4':'base.feature_1.3.conv.1.1',
    'inverted_residual_blocks.0.1.conv.6':'base.feature_1.3.conv.2',
    'inverted_residual_blocks.0.1.conv.7':'base.feature_1.3.conv.3',

    'inverted_residual_blocks.1.0.conv.0':'base.feature_2.0.conv.0.0',
    'inverted_residual_blocks.1.0.conv.1':'base.feature_2.0.conv.0.1',
    'inverted_residual_blocks.1.0.conv.3':'base.feature_2.0.conv.1.0',
    'inverted_residual_blocks.1.0.conv.4':'base.feature_2.0.conv.1.1',
    'inverted_residual_blocks.1.0.conv.6':'base.feature_2.0.conv.2',
    'inverted_residual_blocks.1.0.conv.7':'base.feature_2.0.conv.3',

    'inverted_residual_blocks.1.1.conv.0':'base.feature_2.1.conv.0.0',
    'inverted_residual_blocks.1.1.conv.1':'base.feature_2.1.conv.0.1',
    'inverted_residual_blocks.1.1.conv.3':'base.feature_2.1.conv.1.0',
    'inverted_residual_blocks.1.1.conv.4':'base.feature_2.1.conv.1.1',
    'inverted_residual_blocks.1.1.conv.6':'base.feature_2.1.conv.2',
    'inverted_residual_blocks.1.1.conv.7':'base.feature_2.1.conv.3',

    'inverted_residual_blocks.1.2.conv.0':'base.feature_2.2.conv.0.0',
    'inverted_residual_blocks.1.2.conv.1':'base.feature_2.2.conv.0.1',
    'inverted_residual_blocks.1.2.conv.3':'base.feature_2.2.conv.1.0',
    'inverted_residual_blocks.1.2.conv.4':'base.feature_2.2.conv.1.1',
    'inverted_residual_blocks.1.2.conv.6':'base.feature_2.2.conv.2',
    'inverted_residual_blocks.1.2.conv.7':'base.feature_2.2.conv.3',

    'inverted_residual_blocks.2.0.conv.0':'base.feature_4.0.conv.0.0',
    'inverted_residual_blocks.2.0.conv.1':'base.feature_4.0.conv.0.1',
    'inverted_residual_blocks.2.0.conv.3':'base.feature_4.0.conv.1.0',
    'inverted_residual_blocks.2.0.conv.4':'base.feature_4.0.conv.1.1',
    'inverted_residual_blocks.2.0.conv.6':'base.feature_4.0.conv.2',
    'inverted_residual_blocks.2.0.conv.7':'base.feature_4.0.conv.3',
    
    'inverted_residual_blocks.2.1.conv.0':'base.feature_4.1.conv.0.0',
    'inverted_residual_blocks.2.1.conv.1':'base.feature_4.1.conv.0.1',
    'inverted_residual_blocks.2.1.conv.3':'base.feature_4.1.conv.1.0',
    'inverted_residual_blocks.2.1.conv.4':'base.feature_4.1.conv.1.1',
    'inverted_residual_blocks.2.1.conv.6':'base.feature_4.1.conv.2',
    'inverted_residual_blocks.2.1.conv.7':'base.feature_4.1.conv.3',

    'inverted_residual_blocks.2.2.conv.0':'base.feature_4.2.conv.0.0',
    'inverted_residual_blocks.2.2.conv.1':'base.feature_4.2.conv.0.1',
    'inverted_residual_blocks.2.2.conv.3':'base.feature_4.2.conv.1.0',
    'inverted_residual_blocks.2.2.conv.4':'base.feature_4.2.conv.1.1',
    'inverted_residual_blocks.2.2.conv.6':'base.feature_4.2.conv.2',
    'inverted_residual_blocks.2.2.conv.7':'base.feature_4.2.conv.3',

    'inverted_residual_blocks.2.3.conv.0':'base.feature_4.3.conv.0.0',
    'inverted_residual_blocks.2.3.conv.1':'base.feature_4.3.conv.0.1',
    'inverted_residual_blocks.2.3.conv.3':'base.feature_4.3.conv.1.0',
    'inverted_residual_blocks.2.3.conv.4':'base.feature_4.3.conv.1.1',
    'inverted_residual_blocks.2.3.conv.6':'base.feature_4.3.conv.2',
    'inverted_residual_blocks.2.3.conv.7':'base.feature_4.3.conv.3',

    'inverted_residual_blocks.3.0.conv.0':'base.feature_4.4.conv.0.0',
    'inverted_residual_blocks.3.0.conv.1':'base.feature_4.4.conv.0.1',
    'inverted_residual_blocks.3.0.conv.3':'base.feature_4.4.conv.1.0',
    'inverted_residual_blocks.3.0.conv.4':'base.feature_4.4.conv.1.1',
    'inverted_residual_blocks.3.0.conv.6':'base.feature_4.4.conv.2',
    'inverted_residual_blocks.3.0.conv.7':'base.feature_4.4.conv.3',

    'inverted_residual_blocks.3.1.conv.0':'base.feature_4.5.conv.0.0',
    'inverted_residual_blocks.3.1.conv.1':'base.feature_4.5.conv.0.1',
    'inverted_residual_blocks.3.1.conv.3':'base.feature_4.5.conv.1.0',
    'inverted_residual_blocks.3.1.conv.4':'base.feature_4.5.conv.1.1',
    'inverted_residual_blocks.3.1.conv.6':'base.feature_4.5.conv.2',
    'inverted_residual_blocks.3.1.conv.7':'base.feature_4.5.conv.3',

    'inverted_residual_blocks.3.2.conv.0':'base.feature_4.6.conv.0.0',
    'inverted_residual_blocks.3.2.conv.1':'base.feature_4.6.conv.0.1',
    'inverted_residual_blocks.3.2.conv.3':'base.feature_4.6.conv.1.0',
    'inverted_residual_blocks.3.2.conv.4':'base.feature_4.6.conv.1.1',
    'inverted_residual_blocks.3.2.conv.6':'base.feature_4.6.conv.2',
    'inverted_residual_blocks.3.2.conv.7':'base.feature_4.6.conv.3',

    'inverted_residual_blocks.4.0.conv.0':'base.feature_6.0.conv.0.0',
    'inverted_residual_blocks.4.0.conv.1':'base.feature_6.0.conv.0.1',
    'inverted_residual_blocks.4.0.conv.3':'base.feature_6.0.conv.1.0',
    'inverted_residual_blocks.4.0.conv.4':'base.feature_6.0.conv.1.1',
    'inverted_residual_blocks.4.0.conv.6':'base.feature_6.0.conv.2',
    'inverted_residual_blocks.4.0.conv.7':'base.feature_6.0.conv.3',

    'inverted_residual_blocks.4.1.conv.0':'base.feature_6.1.conv.0.0',
    'inverted_residual_blocks.4.1.conv.1':'base.feature_6.1.conv.0.1',
    'inverted_residual_blocks.4.1.conv.3':'base.feature_6.1.conv.1.0',
    'inverted_residual_blocks.4.1.conv.4':'base.feature_6.1.conv.1.1',
    'inverted_residual_blocks.4.1.conv.6':'base.feature_6.1.conv.2',
    'inverted_residual_blocks.4.1.conv.7':'base.feature_6.1.conv.3',

    'inverted_residual_blocks.4.2.conv.0':'base.feature_6.2.conv.0.0',
    'inverted_residual_blocks.4.2.conv.1':'base.feature_6.2.conv.0.1',
    'inverted_residual_blocks.4.2.conv.3':'base.feature_6.2.conv.1.0',
    'inverted_residual_blocks.4.2.conv.4':'base.feature_6.2.conv.1.1',
    'inverted_residual_blocks.4.2.conv.6':'base.feature_6.2.conv.2',
    'inverted_residual_blocks.4.2.conv.7':'base.feature_6.2.conv.3',

    'inverted_residual_blocks.5.0.conv.0':'base.feature_6.3.conv.0.0',
    'inverted_residual_blocks.5.0.conv.1':'base.feature_6.3.conv.0.1',
    'inverted_residual_blocks.5.0.conv.3':'base.feature_6.3.conv.1.0',
    'inverted_residual_blocks.5.0.conv.4':'base.feature_6.3.conv.1.1',
    'inverted_residual_blocks.5.0.conv.6':'base.feature_6.3.conv.2',
    'inverted_residual_blocks.5.0.conv.7':'base.feature_6.3.conv.3',
}
model1 = Centerface(24)
model1.train()
s1 = model1.base_fpn.state_dict()

model2 = torch.load('models/model_best.pth', map_location='cpu')
s2 = model2['state_dict']

s3 = {}

for key in s1.keys():
    k = None
    for rep_key in replace_dict.keys():
        if rep_key in key:
            k = rep_key
            break
    if k is None:
        raise KeyError(f'cant find key {key} in replace_dict')
    s2_key = key.replace(k, replace_dict[k])
    s3[key] = s2[s2_key]

model1.base_fpn.load_state_dict(s3)
torch.save(model1.base_fpn.state_dict(), 'models/base_fpn_pretrained.pt')
print('state dict is ready')