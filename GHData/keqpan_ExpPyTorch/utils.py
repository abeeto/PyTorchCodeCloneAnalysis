import datetime
import sys
import torch

def printoneline(*argv):
    s = ''
    for arg in argv: s += str(arg) + ' '
    s = s[:-1]
    sys.stdout.write('\r'+s)
    sys.stdout.flush()

def dt():
    return datetime.datetime.now().strftime('%H:%M:%S')

def freeze_model(model):
#     model.train(False)
    model.eval()
    for params in model.parameters():
        params.requires_grad = False

def unfreeze_model(model):
    for params in model.parameters():
        params.requires_grad = True

def KFold(n=6000, n_folds=10, shuffle=False):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[i*n//n_folds:(i+1)*n//n_folds]
        train = list(set(base)-set(test))
        folds.append([train,test])
    return folds

def save_model(model, model_ckpt_path, multigpu_mode=False):
    if multigpu_mode:
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    torch.save(model_state, model_ckpt_path)
    
def load_model(model, model_ckpt_path, multigpu_mode, use_cuda=True):
    model_state_dict = torch.load(model_ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(model_state_dict)
    if multigpu_mode:
        model = torch.nn.DataParallel(model)
    if use_cuda:
        model = model.cuda()
    return model