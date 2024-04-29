import torch
import copy

def save_state(pt, model, optim, num_epochs, best_model, best_metrics):
    copy_model = copy.deepcopy(model)
    copy_best_model = copy.deepcopy(best_model)
    def unpack_model(copy_model):
        if isinstance(copy_model, torch.nn.DataParallel):
            copy_model = copy_model.module
        return copy_model.cpu()
    torch.save({'epoch': num_epochs,
                'model_state': unpack_model(copy_model).state_dict(),
                'optim_state': optim.state_dict(),
                'best_model_state': unpack_model(copy_best_model).state_dict(),
                'best_metrics': best_metrics}, pt)

def load_state(pt, model, optim=None, best_model=None, use_best=False):

    state = torch.load(pt)
    if optim is not None:
        optim.load_state_dict(state['optim_state'])

    def load_pack_model(m, state_dict):
        is_parallel = False
        is_cuda = False
        if isinstance(m, torch.nn.DataParallel):
            m = m.module
            is_parallel = True
            is_cuda = True
        elif next(model.parameters()).is_cuda:
            is_cuda = True
        m.cpu().load_state_dict(state_dict)
        if is_cuda:
            m = m.cuda()
        if is_parallel:
            m = torch.nn.DataParallel(m)
        return m
    if use_best:
        model = load_pack_model(model, state['best_model_state'])
    else:
        model = load_pack_model(model, state['model_state'])
    if best_model is not None:
        best_model = load_pack_model(best_model, state['best_model_state'])
    return state['epoch'], model, optim, best_model, state['best_metrics']
