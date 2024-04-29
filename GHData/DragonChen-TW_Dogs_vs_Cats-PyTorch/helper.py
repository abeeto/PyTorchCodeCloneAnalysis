def set_param_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False
