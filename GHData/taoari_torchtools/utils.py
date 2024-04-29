from ._summary import print_summary
from ._viz import plot_network

def named_leaf_modules(mod):
    for name, module in mod.named_modules():
        if not module._modules:
            yield name, module

def named_modules(mod, leaf_only=False):
    for name, module in mod.named_modules():
        if (not leaf_only) or (leaf_only and not module._modules):
            yield name, module

def unregister_all_hooks(module):
    for name, mod in module.named_modules():
        if mod._forward_hooks:
            mod._forward_hooks.clear()
        if mod._forward_pre_hooks:
            mod._forward_pre_hooks.clear()
        if mod._backward_hooks:
            mod._backward_hooks.clear()

def _unwrap_model(model):
    # if isinstance(model, (nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
    if hasattr(model, 'module'):
        model = model.module
    return model

def _get_device(model):
    __first_param = next(model.parameters(), None)
    return torch.device('cpu') if __first_param is None else __first_param.device


############################################################
# Non-Torch
############################################################

def imread_url(url):
    from PIL import Image
    import requests
    from io import BytesIO
    
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img
