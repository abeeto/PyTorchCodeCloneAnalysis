#%% Imports
import torch
from torchvision import transforms

import ct_color_torch.utils as utils
from ct_color_torch.transformer_net import TransformerNet

#%%Vars
cuda = 1
content_image = 'ct_color_torch\Tomografia_Cx6_T2.PNG'
content_scale = None
style_num = 12
model = 'ct_color_torch\output\epoch_15_Wed_Dec__4_140019_2019_100000_10000000000.model'
style_ids = range(style_num)
output_image = 'test'

device = torch.device("cuda" if cuda == 1 else "cpu")

#%% Load content image
content_image = utils.load_image(content_image, scale=content_scale)
content_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])
content_image = content_transform(content_image)
content_image = content_image.unsqueeze(0).to(device)

#%% Load model and apply style
for style in style_ids:
    style_id = style
    with torch.no_grad():
        style_model = TransformerNet(style_num=style_num)
        state_dict = torch.load(model).copy()
        for k in list(state_dict.keys()):
            if 'running_mean' in k:
                del state_dict[k]
            elif 'running_var' in k:
                del state_dict[k]


        style_model.load_state_dict(state_dict,strict=False)
        style_model.to(device)
        output = style_model(content_image, style_id = [style_id]).cpu()

    utils.save_image('ct_color_torch\\output\\imgs\\'+output_image+'_style'+str(style_id)+'.jpg', output[0])

  # %%


# %%
